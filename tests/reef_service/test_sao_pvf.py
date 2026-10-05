"""Privileged value function (arXiv:2608.16739) on SAO: the critic-only prefix path.

The contract: a report's ``critic_context`` becomes token ids that only the
critic worker puts in front of its copy of the sample. Without a context the
wire payload and the critic's batch are exactly the plain SAO ones; with one,
each response token keeps its index from the sequence end, which is where
Slime reads values.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from recipes.sao.processor import SAOProcessor
from recipes.sao.recipe import SAORecipe
from recipes.sao.report import SAOReport
from recipes.sao.slime import SaoAlgorithm
from recipes.sao.slime.utils.data_builder import build_sao_rollout_data
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings


def _row(prefix: list[int]) -> list[object]:
    return ["i1", [9, 1, 2, 3], [1, 1, 1], [-0.1, -0.2, -0.3], 0.5, [1, 1, 1], "slime-v3", 1234.5, prefix]


def _payload(rows: list[list[object]]) -> dict:
    return {"samples": rows, "rollout_ids": list(range(len(rows))), "loss": "sao"}


@pytest.mark.unit
def test_rows_without_prefix_build_the_plain_sao_payload() -> None:
    rows = [_row([]), _row([])]
    data = build_sao_rollout_data(_payload(rows), rows, SaoAlgorithm())
    assert "critic_prefix_tokens" not in data


@pytest.mark.unit
def test_rows_with_prefix_attach_it_per_sample() -> None:
    rows = [_row([7, 8]), _row([])]
    data = build_sao_rollout_data(_payload(rows), rows, SaoAlgorithm())
    assert data["critic_prefix_tokens"] == [[7, 8], []]
    # The actor's columns are the plain ones.
    assert data["tokens"] == [[9, 1, 2, 3], [9, 1, 2, 3]]


@pytest.mark.unit
def test_prefix_key_is_declared_on_every_wire_surface_it_needs() -> None:
    spec = SaoAlgorithm()
    key = spec.critic_prefix_key
    assert key in spec.rollout_data_keys
    assert spec.rollout_tensor_dtypes[key] == "long"
    assert key in spec.rollout_log_skip_keys
    assert key not in spec.external_batch_keys  # the actor's microbatches never carry it


@pytest.mark.unit
def test_report_parses_critic_context_from_metadata() -> None:
    parsed = SAOReport.from_dict({"score": 1.0, "metadata": {"critic_context": "answer: -92"}})
    assert parsed.critic_context == "answer: -92"
    assert SAOReport.from_dict({"score": 0.0}).critic_context == ""


@pytest.mark.unit
def test_recipe_requires_tokenizer_for_privileged_value() -> None:
    bindings = runtime_bindings(StubTrainingRuntime())
    with pytest.raises(ValueError, match="tokenizer_path"):
        SAORecipe(**bindings, privileged_value=True)
    assert SAORecipe(**bindings, privileged_value=True, tokenizer_path="/models/x").privileged_value
    assert not SAORecipe(**bindings).privileged_value


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": False, "return_dict": False}
        assert messages == [{"role": "system", "content": messages[0]["content"]}]
        return [100, *(ord(ch) for ch in messages[0]["content"]), 101]


def _processor(privileged_value: bool) -> SAOProcessor:
    processor = object.__new__(SAOProcessor)
    processor._privileged_value = privileged_value
    processor._tokenizer = _Tokenizer() if privileged_value else None
    return processor


def _context(critic_context: str) -> SimpleNamespace:
    return SimpleNamespace(
        parsed_report=SAOReport(score=1.0, critic_context=critic_context),
        report=SimpleNamespace(agent_record_id="r1"),
    )


@pytest.mark.unit
def test_context_is_used_only_when_privileged_value_is_on() -> None:
    assert _processor(False)._critic_context(_context("")) == ""
    assert _processor(False)._critic_context(_context("ab")) == ""  # off: ignored, as upstream SAO
    assert _processor(True)._critic_context(_context("ab")) == "ab"
    with pytest.raises(ValueError, match="has no critic_context"):
        _processor(True)._critic_context(_context(" "))


@pytest.mark.unit
def test_critic_prefix_is_the_context_as_a_system_turn() -> None:
    assert _processor(True).critic_prefix_tokens("ab") == [100, 97, 98, 101]


@pytest.mark.unit
def test_prepend_keeps_every_response_token_at_its_index_from_the_end(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from megatron.core import mpu

    from reef.train.slime_backend.reef_adapters.megatron import train_actor

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    tokens = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([6, 7, 8])]
    response_lengths = [2, 1]
    rollout_data = {"tokens": list(tokens), "total_lengths": [5, 3], "response_lengths": response_lengths}
    prefixes = [torch.tensor([90, 91, 92]), torch.tensor([], dtype=torch.long)]

    train_actor._prepend_critic_prefix(rollout_data, prefixes, "cpu")

    assert rollout_data["total_lengths"] == [8, 3]
    assert rollout_data["tokens"][0].tolist() == [90, 91, 92, 1, 2, 3, 4, 5]
    assert rollout_data["tokens"][1].tolist() == [6, 7, 8]
    # Slime's get_responses slices [total - response - 1, total - 1) of each sample's positions:
    # the positions that predict the response tokens are the same tokens as before, shifted by the prefix.
    for before, after, total_before, total_after, response in zip(
        tokens, rollout_data["tokens"], [5, 3], rollout_data["total_lengths"], response_lengths, strict=True
    ):
        assert torch.equal(
            before[total_before - response - 1 : total_before - 1],
            after[total_after - response - 1 : total_after - 1],
        )


@pytest.mark.unit
def test_prepend_refuses_context_parallelism(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from megatron.core import mpu

    from reef.train.slime_backend.reef_adapters.megatron import train_actor

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    rollout_data = {"tokens": [torch.tensor([1, 2])], "total_lengths": [2], "response_lengths": [1]}
    with pytest.raises(RuntimeError, match="context parallel size 1"):
        train_actor._prepend_critic_prefix(rollout_data, [torch.tensor([9])], "cpu")
