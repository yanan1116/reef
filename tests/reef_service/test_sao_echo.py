"""Tests for SAO's optional ECHO term (environment-observation cross-entropy, ``recipes.sao.slime.echo``).

They need CPU ``torch`` only: Slime's loss module is replaced by a small stand-in with the same three names ECHO
wraps or calls (``get_log_probs_and_entropy``, ``policy_loss_function``, ``get_sum_of_sample_mean``).
"""

from __future__ import annotations

import sys
import types
from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")

from recipes.sao.slime import SaoAlgorithm, SaoSettings  # noqa: E402
from recipes.sao.slime.echo import (  # noqa: E402
    EchoRules,
    echo_coefficient,
    echo_cross_entropy,
    install_echo,
    observation_masks,
)
from recipes.sao.slime.objective import sao_actor_init  # noqa: E402

OPEN, CLOSE = 90, 91
# A toy vocabulary: 1/2 are template tokens, 3 is a newline, 40-44 spell an action, 20-29 an observation.
WORDS = {
    1: "<|im_end|>",
    2: "<|im_start|>assistant",
    3: "\n",
    7: "x",
    8: "y",
    20: "ok",
    21: " done",
    22: '{"error": ',
    23: "budget",
    24: "[truncated]",
    25: "Docs:",
    40: "print(",
    41: "apis.api_docs.show(",
    42: ")",
    OPEN: "<tr>",
    CLOSE: "</tr>",
}


def decode(token_ids: list[int]) -> str:
    return "".join(WORDS.get(token, "?") for token in token_ids)


def sum_of_sample_mean_over(masks: list[torch.Tensor]):
    """Slime's per-sample masked mean, summed over samples (no context parallelism)."""

    def reduce(values: torch.Tensor) -> torch.Tensor:
        total, start = values.new_zeros(()), 0
        for mask in masks:
            row = values[start : start + mask.numel()]
            total = total + (row * mask).sum() / mask.sum().clamp_min(1)
            start += mask.numel()
        return total

    return reduce


def test_tool_span_keeps_only_text_inside_tool_response_blocks() -> None:
    # prompt [5, 6]; response: action 7, <im_end>-like 1, OPEN, obs 20 21, CLOSE, scaffold 2, action 8
    tokens = torch.tensor([5, 6, 7, 1, OPEN, 20, 21, CLOSE, 2, 8])
    action = torch.tensor([1, 0, 0, 0, 0, 0, 0, 1])
    (mask,) = observation_masks([tokens], [8], [action], "tool", OPEN, CLOSE, decode)
    assert mask.tolist() == [0, 0, 0, 1, 1, 0, 0, 0]


def test_all_span_takes_every_non_action_token() -> None:
    tokens = torch.tensor([5, 6, 7, 1, OPEN, 20, 21, CLOSE, 2, 8])
    action = torch.tensor([1, 0, 0, 0, 0, 0, 0, 1])
    (mask,) = observation_masks([tokens], [8], [action], "all")
    assert mask.tolist() == [0, 1, 1, 1, 1, 1, 1, 0]


def test_action_tokens_inside_a_tool_block_are_never_predicted() -> None:
    tokens = torch.tensor([5, OPEN, 20, 21, CLOSE])
    action = torch.tensor([0, 1, 0, 0])
    (mask,) = observation_masks([tokens], [4], [action], "tool", OPEN, CLOSE, decode)
    assert mask.tolist() == [0, 0, 1, 0]


def test_template_newlines_around_the_content_are_left_out() -> None:
    # response: action 7, OPEN, "\n", ok, done, "\n", CLOSE, action 8
    tokens = torch.tensor([5, 7, OPEN, 3, 20, 21, 3, CLOSE, 8])
    action = torch.tensor([1, 0, 0, 0, 0, 0, 0, 1])
    (mask,) = observation_masks([tokens], [8], [action], "tool", OPEN, CLOSE, decode)
    assert mask.tolist() == [0, 0, 0, 1, 1, 0, 0, 0]


def test_rules_skip_harness_messages_and_drop_the_truncation_marker() -> None:
    rules = EchoRules(skip_response_prefixes=('{"error": ',), drop_response_suffixes=("[truncated]",))
    # two tool responses: a harness error message, then an output that ends with the harness's truncation line
    tokens = torch.tensor([5, 7, OPEN, 22, 23, CLOSE, 8, OPEN, 20, 21, 3, 24, CLOSE, 7])
    action = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    counts: dict[str, int] = {}
    (mask,) = observation_masks([tokens], [13], [action], "tool", OPEN, CLOSE, decode, rules, counts)
    assert mask.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    assert counts == {"skipped": 1, "kept": 1}


def test_rules_can_skip_responses_by_the_action_that_produced_them() -> None:
    rules = EchoRules(skip_after_action=("api_docs",))
    # action 1 prints API docs; action 2 prints data
    tokens = torch.tensor([5, 41, 42, OPEN, 25, CLOSE, 40, 42, OPEN, 20, CLOSE, 8])
    action = torch.tensor([1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    (mask,) = observation_masks([tokens], [11], [action], "tool", OPEN, CLOSE, decode, rules)
    assert mask.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]


def test_rules_can_keep_only_responses_with_given_prefixes() -> None:
    # errors-only ECHO: of a data output and an error output, only the error is kept
    rules = EchoRules(keep_response_prefixes=('{"error": ',))
    tokens = torch.tensor([5, 7, OPEN, 20, 21, CLOSE, 8, OPEN, 22, 23, CLOSE, 7])
    action = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    counts: dict[str, int] = {}
    (mask,) = observation_masks([tokens], [11], [action], "tool", OPEN, CLOSE, decode, rules, counts)
    assert mask.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert counts == {"skipped": 1, "kept": 1}


def test_rules_can_keep_only_the_last_line_of_each_response() -> None:
    # an error response of two lines: quoted code, then the exception line; only the exception line is kept
    rules = EchoRules(keep_response_prefixes=("ok",), keep_last_line=True)
    tokens = torch.tensor([5, 7, OPEN, 3, 20, 21, 3, 22, 23, 3, CLOSE, 7])
    action = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    (mask,) = observation_masks([tokens], [11], [action], "tool", OPEN, CLOSE, decode, rules)
    assert mask.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    # a one-line response is kept whole
    tokens = torch.tensor([5, 7, OPEN, 20, 21, CLOSE, 7])
    action = torch.tensor([1, 0, 0, 0, 0, 1])
    (mask,) = observation_masks([tokens], [6], [action], "tool", OPEN, CLOSE, decode, rules)
    assert mask.tolist() == [0, 0, 1, 1, 0, 0]


def test_rules_file_loads_and_rejects_unknown_keys(tmp_path) -> None:
    good = tmp_path / "rules.json"
    good.write_text('{"description": "x", "skip_after_action": ["api_docs"], "drop_response_suffixes": ["[t]"]}')
    assert EchoRules.load(str(good)) == EchoRules(skip_after_action=("api_docs",), drop_response_suffixes=("[t]",))
    assert EchoRules.load(None) == EchoRules()
    last = tmp_path / "last.json"
    last.write_text('{"keep_response_prefixes": ["Execution failed."], "keep_last_line": true}')
    assert EchoRules.load(str(last)) == EchoRules(keep_response_prefixes=("Execution failed.",), keep_last_line=True)
    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"keep_last_line": "yes"}')
    with pytest.raises(ValueError, match="keep_last_line"):
        EchoRules.load(str(wrong))
    bad = tmp_path / "bad.json"
    bad.write_text('{"skip_everything": true}')
    with pytest.raises(ValueError, match="keys"):
        EchoRules.load(str(bad))


def test_masks_reject_misaligned_action_masks_and_bad_span() -> None:
    with pytest.raises(ValueError, match="action mask"):
        observation_masks([torch.tensor([1, 2, 3])], [2], [torch.tensor([1, 0, 0])], "all")
    with pytest.raises(ValueError, match="span"):
        observation_masks([torch.tensor([1, 2])], [1], [torch.tensor([1])], "observations")
    with pytest.raises(ValueError, match="tool_response"):
        observation_masks([torch.tensor([1, 2])], [1], [torch.tensor([1])], "tool", OPEN, CLOSE)


def test_cross_entropy_is_the_per_sample_mean_of_negative_log_probs() -> None:
    log_probs = [torch.tensor([-1.0, -2.0, -4.0]), torch.tensor([-3.0, -5.0])]
    masks = [torch.tensor([0.0, 1.0, 1.0]), torch.tensor([0.0, 0.0])]
    ce = echo_cross_entropy(log_probs, masks, sum_of_sample_mean_over(masks))
    # sample 1: mean(2, 4) = 3; sample 2 has no observation token and adds 0
    assert ce.item() == pytest.approx(3.0)


def test_options_default_off_and_parse_and_reach_args() -> None:
    algorithm = SaoAlgorithm()
    settings, rest = algorithm.parse_specific_options(
        ["--sao-echo-coef", "0.05", "--sao-echo-span", "all", "--sao-echo-rules", "/r.json", "--x"]
    )
    assert (settings.echo_coef, settings.echo_span, settings.echo_rules, rest) == (0.05, "all", "/r.json", ["--x"])
    assert (SaoSettings().echo_coef, SaoSettings().echo_span, SaoSettings().echo_rules) == (0.0, "tool", None)
    args = Namespace()
    algorithm.apply_driver_options(args, settings)
    assert (args.sao_echo_coef, args.sao_echo_span, args.sao_echo_rules) == (0.05, "all", "/r.json")


def _valid_sao_args(**overrides: object) -> Namespace:
    values = dict(
        use_critic=True,
        eps_clip=0.3,
        eps_clip_high=5.0,
        num_critic_only_steps=0,
        loss_type="policy_loss",
        sao_echo_coef=0.05,
        sao_echo_span="tool",
        context_parallel_size=1,
    )
    values.update(overrides)
    return Namespace(**values)


def test_validation_rejects_negative_coef_context_parallelism_and_other_loss_types() -> None:
    algorithm = SaoAlgorithm()
    algorithm.validate_specific_args(_valid_sao_args(), "sao")
    for overrides, message in [
        ({"sao_echo_coef": -0.1}, "sao-echo-coef"),
        ({"context_parallel_size": 2}, "context-parallel-size"),
        ({"loss_type": "custom_loss"}, "policy_loss"),
        ({"sao_echo_rules": "/no/such/rules.json"}, "not a file"),
    ]:
        with pytest.raises(RuntimeError, match=message):
            algorithm.validate_specific_args(_valid_sao_args(**overrides), "sao")
    algorithm.validate_specific_args(_valid_sao_args(sao_echo_coef=0.0, context_parallel_size=2), "sao")


@pytest.fixture
def fake_slime_loss(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A stand-in ``slime.backends.megatron_utils.loss`` whose policy loss calls ``get_log_probs_and_entropy``."""
    module = types.ModuleType("slime.backends.megatron_utils.loss")

    def get_log_probs_and_entropy(logits, **kwargs):
        return torch.empty((0,)), {"log_probs": [row * logits for row in kwargs["rows"]]}

    def policy_loss_function(args, batch, logits, sum_of_sample_mean):
        _, result = module.get_log_probs_and_entropy(logits, rows=batch["rows"])
        pg = torch.cat(result["log_probs"]).sum() * 0.0 + 1.5
        return pg, {"loss": pg.detach(), "pg_loss": pg.detach()}

    def get_sum_of_sample_mean(total_lengths, response_lengths, masks, denoms, per_token):
        return sum_of_sample_mean_over(masks)

    module.get_log_probs_and_entropy = get_log_probs_and_entropy
    module.policy_loss_function = policy_loss_function
    module.get_sum_of_sample_mean = get_sum_of_sample_mean
    for name in ("slime", "slime.backends", "slime.backends.megatron_utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.loss", module)
    sys.modules["slime.backends.megatron_utils"].loss = module
    return module


def test_actor_hook_leaves_slime_untouched_when_off(fake_slime_loss: types.ModuleType) -> None:
    before = (fake_slime_loss.policy_loss_function, fake_slime_loss.get_log_probs_and_entropy)
    sao_actor_init(types.SimpleNamespace(args=Namespace(sao_echo_coef=0.0)))
    assert (fake_slime_loss.policy_loss_function, fake_slime_loss.get_log_probs_and_entropy) == before


def test_installed_echo_adds_coef_times_cross_entropy_once(fake_slime_loss: types.ModuleType) -> None:
    args = Namespace(sao_echo_coef=0.5, sao_echo_span="all", calculate_per_token_loss=False, hf_checkpoint="unused")
    sao_actor_init(types.SimpleNamespace(args=args))
    sao_actor_init(types.SimpleNamespace(args=args))  # idempotent
    batch = {
        "rows": [torch.tensor([-1.0, -2.0, -4.0])],
        "unconcat_tokens": [torch.tensor([5, 7, 1, 8])],
        "response_lengths": [3],
        "total_lengths": [4],
        "action_masks": [torch.tensor([1, 0, 1])],
    }
    total, log = fake_slime_loss.policy_loss_function(args, batch, torch.tensor(1.0), None)
    # pg 1.5 plus 0.5 * CE on the single observation token (-log p = 2)
    assert total.item() == pytest.approx(2.5)
    assert log["echo_ce"].item() == pytest.approx(2.0)
    assert log["loss"].item() == pytest.approx(2.5)
    assert log["pg_loss"].item() == pytest.approx(1.5)
    assert log["echo_observation_tokens"].item() == 1


def test_target_ratio_keeps_the_term_at_that_share_of_the_policy_loss() -> None:
    args = Namespace(sao_echo_coef=0.05, sao_echo_target_ratio=0.2, sao_echo_ema=0.5)
    balance: dict[str, float] = {}
    coef = echo_coefficient(args, balance, torch.tensor(-0.15), torch.tensor(6.0))
    assert coef * 6.0 == pytest.approx(0.2 * 0.15)
    # averages move halfway (ema 0.5): |pg| 0.15 -> 0.125, ce 6 -> 4
    coef = echo_coefficient(args, balance, torch.tensor(0.1), torch.tensor(2.0))
    assert coef == pytest.approx(0.2 * 0.125 / 4.0)
    # the fixed coef caps the weight
    assert echo_coefficient(args, {}, torch.tensor(10.0), torch.tensor(0.1)) == 0.05
    assert echo_coefficient(Namespace(sao_echo_coef=0.01), {}, torch.tensor(1.0), torch.tensor(1.0)) == 0.01


def test_installed_echo_logs_the_adaptive_weight_and_ratio(fake_slime_loss: types.ModuleType) -> None:
    args = Namespace(
        sao_echo_coef=0.5,
        sao_echo_target_ratio=0.2,
        sao_echo_ema=0.9,
        sao_echo_span="all",
        calculate_per_token_loss=False,
        hf_checkpoint="unused",
    )
    sao_actor_init(types.SimpleNamespace(args=args))
    batch = {
        "rows": [torch.tensor([-1.0, -2.0, -4.0])],
        "unconcat_tokens": [torch.tensor([5, 7, 1, 8])],
        "response_lengths": [3],
        "total_lengths": [4],
        "action_masks": [torch.tensor([1, 0, 1])],
    }
    total, log = fake_slime_loss.policy_loss_function(args, batch, torch.tensor(1.0), None)
    # pg 1.5, CE 2: coef = 0.2 * 1.5 / 2 = 0.15, term 0.3 = 20% of the policy loss
    assert log["echo_coef"].item() == pytest.approx(0.15)
    assert log["echo_term_ratio"].item() == pytest.approx(0.2)
    assert total.item() == pytest.approx(1.8)
