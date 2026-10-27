"""ECHO auxiliary objective for the SAO actor (Environment Cross-entropy Hybrid Objective, arXiv:2605.24517).

The actor also learns to predict the environment's replies: a cross-entropy on observation tokens, the tool results
that arrive between the model's actions, is added to SAO's policy loss::

    loss = L_SAO + coef * L_env,  L_env = mean over samples of the mean of -log p(o_t | x_<t) over observation tokens

Off unless ``--sao-echo-coef`` is positive. Only the actor's loss changes: the critic, the skip-observation GAE and
SAO's ratio calibration see the same tensors as without ECHO, and observation tokens still carry no advantage.

The term reuses the log-probabilities Slime's policy loss already computes on the whole response region (observation
positions included; ``loss_mask`` only selects which of them the policy gradient uses), so it adds no forward pass
and no second vocabulary-sized tensor. ``--sao-echo-span tool`` keeps only the text inside the chat template's
``<tool_response>`` ... ``</tool_response>`` blocks; ``all`` takes every non-action token after the first prompt,
including chat-template scaffolding and realigned re-renders.
"""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from recipes.sao.slime import ECHO_SPANS

TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


@dataclass(frozen=True)
class EchoRules:
    """Which tool responses, and which of their tokens, ECHO leaves out (``--sao-echo-rules``, a JSON file).

    ECHO predicts what the environment returned, not text the harness wrote (arXiv:2605.24517, Section 3.2: format
    warnings are left out because the model memorizes them and they carry no lasting signal). An environment's rules
    name that text:

    - ``skip_after_action``: skip a tool response when the action text that produced it contains one of these
      strings, e.g. ``api_docs`` for AppWorld's API documentation (static text, not the consequence of an action);
    - ``skip_response_prefixes``: skip a tool response whose text starts with one of these (a message the harness
      wrote instead of the environment's output);
    - ``drop_response_suffixes``: leave out these markers when a tool response ends with one (e.g. a truncation note
      the harness appended);
    - ``keep_response_prefixes``: when given, keep only tool responses whose text starts with one of these, e.g.
      AppWorld's ``Execution failed.`` for errors-only ECHO (user, 2026-10-24). Empty (default): no such filter;
    - ``keep_last_line``: predict only the last non-empty line of each kept response, e.g. the exception line of a
      traceback, without the quoted source code and frame lines above it (user, 2026-10-25). Default false.
    """

    skip_after_action: tuple[str, ...] = ()
    skip_response_prefixes: tuple[str, ...] = ()
    drop_response_suffixes: tuple[str, ...] = ()
    keep_response_prefixes: tuple[str, ...] = ()
    keep_last_line: bool = False

    @classmethod
    def load(cls, path: str | None) -> EchoRules:
        if not path:
            return cls()
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        keys = ("skip_after_action", "skip_response_prefixes", "drop_response_suffixes", "keep_response_prefixes")
        known = {*keys, "keep_last_line", "description"}
        if not isinstance(data, dict) or set(data) - known:
            raise ValueError(f"ECHO rules {path} must be an object with keys from {sorted(known)}")
        values = {}
        for key in keys:
            items = data.get(key, [])
            if not isinstance(items, list) or not all(isinstance(item, str) and item for item in items):
                raise ValueError(f"ECHO rules {path}: {key} must be a list of non-empty strings")
            values[key] = tuple(items)
        keep_last_line = data.get("keep_last_line", False)
        if not isinstance(keep_last_line, bool):
            raise ValueError(f"ECHO rules {path}: keep_last_line must be true or false")
        return cls(**values, keep_last_line=keep_last_line)


def tool_response_spans(tokens: Sequence[int], open_id: int, close_id: int) -> list[tuple[int, int]]:
    """``(start, end)`` of every tool response's content: the tokens strictly between ``open_id`` and ``close_id``."""
    spans, start = [], None
    for index, token in enumerate(tokens):
        if token == open_id:
            start = index + 1
        elif token == close_id and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(tokens)))
    return spans


def _content_bounds(
    tokens: Sequence[int], start: int, end: int, decode: Callable[[list[int]], str], rules: EchoRules
) -> tuple[int, int]:
    """The response's content without the template's whitespace around it and without a dropped trailing marker."""

    def trim(start: int, end: int) -> tuple[int, int]:
        while start < end and not decode([tokens[start]]).strip():
            start += 1
        while end > start and not decode([tokens[end - 1]]).strip():
            end -= 1
        return start, end

    start, end = trim(start, end)
    for suffix in rules.drop_response_suffixes:
        if end > start and decode(list(tokens[start:end])).rstrip().endswith(suffix):
            cut = end - 1
            while cut > start and suffix not in decode(list(tokens[cut:end])):
                cut -= 1
            start, end = trim(start, cut)
    return start, end


def _last_line_start(tokens: Sequence[int], start: int, end: int, decode: Callable[[list[int]], str]) -> int:
    """The first token of the content's last non-empty line (the token that holds the line's first character)."""
    pieces = [decode([token]) for token in tokens[start:end]]
    line_start = "".join(pieces).rstrip().rfind("\n") + 1
    offset = 0
    for index, piece in enumerate(pieces):
        if offset + len(piece) > line_start:
            start = start + index
            break
        offset += len(piece)
    while start < end and not decode([tokens[start]]).strip():
        start += 1
    return start


def observation_masks(
    unconcat_tokens: Sequence[torch.Tensor],
    response_lengths: Sequence[int],
    action_masks: Sequence[torch.Tensor],
    span: str,
    open_id: int | None = None,
    close_id: int | None = None,
    decode: Callable[[list[int]], str] | None = None,
    rules: EchoRules | None = None,
    counts: dict[str, int] | None = None,
) -> list[torch.Tensor]:
    """Per-sample float masks over the response region marking the tokens ECHO predicts.

    ``unconcat_tokens[i]`` is the whole sample (prompt then response) and ``action_masks[i]`` covers its last
    ``response_lengths[i]`` tokens, 1 on model-generated tokens. ``span == "all"`` takes every non-action token.
    ``span == "tool"`` takes, for each tool response in the response region, its content between ``open_id`` and
    ``close_id`` without the template's surrounding whitespace, minus what ``rules`` leave out (``decode`` turns token
    ids into text for that). ``counts``, when given, receives ``kept`` and ``skipped`` tool-response totals.
    """
    if span not in ECHO_SPANS:
        raise ValueError(f"ECHO span must be one of {ECHO_SPANS}, got {span!r}")
    if span == "tool" and (open_id is None or close_id is None or decode is None):
        raise ValueError("ECHO span 'tool' needs the <tool_response> open and close token ids and a decoder")
    rules = rules or EchoRules()
    masks = []
    for tokens, response_length, action_mask in zip(unconcat_tokens, response_lengths, action_masks, strict=True):
        response_length = int(response_length)
        if action_mask.numel() != response_length:
            raise ValueError(f"action mask has {action_mask.numel()} tokens for a {response_length}-token response")
        observation = action_mask == 0
        if span == "all":
            masks.append(observation.float())
            continue
        ids = tokens.tolist()
        actions = action_mask.tolist()
        offset = len(ids) - response_length
        keep = torch.zeros(response_length, dtype=torch.bool, device=action_mask.device)
        previous_end = offset
        for start, end in tool_response_spans(ids, open_id, close_id):
            if start < offset:
                continue
            action_text = ""
            if rules.skip_after_action:
                action_ids = [ids[i] for i in range(previous_end, start) if actions[i - offset]]
                action_text = decode(action_ids)
            previous_end = end + 1
            content_start, content_end = _content_bounds(ids, start, end, decode, rules)
            needs_text = rules.skip_response_prefixes or rules.keep_response_prefixes
            text = decode(ids[content_start:content_end]) if needs_text else ""
            skipped = (
                any(marker in action_text for marker in rules.skip_after_action)
                or any(text.startswith(prefix) for prefix in rules.skip_response_prefixes)
                or bool(rules.keep_response_prefixes)
                and not any(text.startswith(prefix) for prefix in rules.keep_response_prefixes)
            )
            if counts is not None:
                counts["skipped" if skipped else "kept"] = counts.get("skipped" if skipped else "kept", 0) + 1
            if not skipped and content_end > content_start:
                if rules.keep_last_line:  # after the checks above, which read the whole response
                    content_start = _last_line_start(ids, content_start, content_end, decode)
                keep[content_start - offset : content_end - offset] = True
        masks.append((observation & keep).float())
    return masks


def echo_cross_entropy(
    log_probs: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """``-log p`` on the masked observation tokens, reduced like SAO's policy loss (per-sample mean, then summed).

    ``sum_of_sample_mean`` must be built over ``masks`` (Slime's ``get_sum_of_sample_mean``); a sample without
    observation tokens contributes zero.
    """
    if len(log_probs) != len(masks):
        raise ValueError(f"{len(log_probs)} log-prob rows for {len(masks)} observation masks")
    for row, mask in zip(log_probs, masks, strict=True):
        if row.numel() != mask.numel():
            raise ValueError(f"log-prob row has {row.numel()} tokens, observation mask {mask.numel()}")
    return sum_of_sample_mean(-torch.cat(list(log_probs), dim=0))


def echo_coefficient(
    args: Namespace, balance: dict[str, float], pg_loss: torch.Tensor, cross_entropy: torch.Tensor
) -> float:
    """The weight on ECHO's cross-entropy for this micro-batch.

    Fixed ``--sao-echo-coef`` unless ``--sao-echo-target-ratio`` is positive. Then the weight keeps the ECHO term at
    that fraction of SAO's policy loss (user, 2026-10-23: ECHO's share at 15-20%): ``ratio * EMA(|pg_loss|) /
    EMA(echo_ce)``, both averages over this worker's micro-batches with ``--sao-echo-ema``, capped at
    ``--sao-echo-coef``. Qwen3.5's cross-entropy on AppWorld tool output starts near 6.3 nats/token and falls as it is
    learned, so a fixed weight would first dominate the policy loss and later fade out.
    """
    ratio = float(getattr(args, "sao_echo_target_ratio", 0.0) or 0.0)
    if ratio <= 0:
        return float(args.sao_echo_coef)
    decay = float(getattr(args, "sao_echo_ema", 0.9))
    for key, value in (("pg", float(pg_loss.abs())), ("ce", float(cross_entropy))):
        balance[key] = value if key not in balance else decay * balance[key] + (1 - decay) * value
    if balance["ce"] <= 0:
        return 0.0
    return min(float(args.sao_echo_coef), ratio * balance["pg"] / balance["ce"])


def load_tool_response_tokenizer(tokenizer_path: str) -> tuple[int, int, Callable[[list[int]], str]]:
    """The token ids of ``<tool_response>`` and ``</tool_response>`` and a decoder, from the policy's tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    ids = []
    for text in (TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"ECHO span 'tool' needs {text!r} to be one token in {tokenizer_path}; got {encoded}")
        ids.append(int(encoded[0]))
    return ids[0], ids[1], lambda token_ids: tokenizer.decode(token_ids, skip_special_tokens=False)


def install_echo(args: Namespace) -> None:
    """Add the ECHO term to Slime's policy loss in this actor worker (idempotent).

    Two module-level wrappers in ``slime.backends.megatron_utils.loss``, the place score centering also extends:
    ``get_log_probs_and_entropy`` keeps its per-sample log-probs while the policy loss runs, and
    ``policy_loss_function`` adds ``coef * L_env`` to the loss it returns. Slime's ``loss_function`` reads both names
    at call time.
    """
    from slime.backends.megatron_utils import loss

    if getattr(loss.policy_loss_function, "_reef_sao_echo", False):
        return
    open_id = close_id = decode = None
    if args.sao_echo_span == "tool":
        open_id, close_id, decode = load_tool_response_tokenizer(args.hf_checkpoint)
    rules = EchoRules.load(getattr(args, "sao_echo_rules", None))
    captured: dict[str, Any] = {}
    balance: dict[str, float] = {}
    base_log_probs = loss.get_log_probs_and_entropy
    base_policy_loss = loss.policy_loss_function

    def get_log_probs_and_entropy(*call_args: Any, **call_kwargs: Any) -> Any:
        result = base_log_probs(*call_args, **call_kwargs)
        if captured.get("active"):
            captured["log_probs"] = result[1]["log_probs"]
        return result

    def policy_loss_function(args, batch, logits, sum_of_sample_mean):
        captured.update(active=True, log_probs=None)
        try:
            total, log = base_policy_loss(args, batch, logits, sum_of_sample_mean)
        finally:
            captured["active"] = False
        log_probs = captured.pop("log_probs", None)
        if log_probs is None:
            raise RuntimeError(
                "ECHO found no policy log-probs: Slime's policy loss no longer calls get_log_probs_and_entropy"
            )
        counts: dict[str, int] = {}
        masks = observation_masks(
            batch["unconcat_tokens"],
            batch["response_lengths"],
            batch["action_masks"],
            args.sao_echo_span,
            open_id,
            close_id,
            decode,
            rules,
            counts,
        )
        reduce = loss.get_sum_of_sample_mean(
            batch["total_lengths"], batch["response_lengths"], masks, None, args.calculate_per_token_loss
        )
        cross_entropy = echo_cross_entropy(log_probs, masks, reduce)
        coef = echo_coefficient(args, balance, log["pg_loss"], cross_entropy.detach())
        total = total + coef * cross_entropy
        log = {
            **log,
            "loss": total.detach().clone(),
            "echo_ce": cross_entropy.detach().clone(),
            "echo_coef": torch.tensor(float(coef), device=total.device),
            "echo_term_ratio": (coef * cross_entropy.detach() / log["pg_loss"].abs().clamp_min(1e-8)).clone(),
            "echo_observation_tokens": torch.stack([mask.sum() for mask in masks]).sum().detach(),
            "echo_responses_kept": torch.tensor(float(counts.get("kept", 0)), device=total.device),
            "echo_responses_skipped": torch.tensor(float(counts.get("skipped", 0)), device=total.device),
        }
        return total, log

    marked: Any = policy_loss_function
    marked._reef_sao_echo = True
    loss.get_log_probs_and_entropy = get_log_probs_and_entropy
    loss.policy_loss_function = policy_loss_function


__all__ = [
    "EchoRules",
    "echo_coefficient",
    "echo_cross_entropy",
    "install_echo",
    "load_tool_response_tokenizer",
    "observation_masks",
    "tool_response_spans",
]
