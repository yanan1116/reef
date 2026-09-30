"""The FinQA multi-table v2 environment the rllm multi-table lines used, without rllm.

rllm_multitable/ holds byte-identical copies of rllm's multitable_v2_flow.py and its system
prompt (see rllm_multitable/SOURCE.txt; the SHA256SUMS are checked at import). This module
exposes what the SAO driver and the evaluator need:

  run_episode(chat, task)   multitable_v2_flow.finqa_multitable_v2's loop, turn for turn, with
                            the model call injected: the driver routes it through Reef (one
                            receipt per call), an evaluator can call vLLM directly.
  grade(task, episode)      rllm's finqa_evaluator; for multi_table questions the reward is the
                            rubric score in [0, 1] and is_correct is score >= 0.9.

The flow's own helpers do the work (public-question builder, tool execution with its error
text, 8000-char truncation, transcript-size check, unparsed-tool-call detection) and its
constants set the budgets. The in-conversation system messages are written out below and
checked at import against the vendored flow's string literals.

One deliberate difference (SOURCE.txt): the final-synthesis calls keep the tool list, so
every call renders the same system prompt and Reef can assemble the episode into one sample.

Training budget (run_episode(..., token_budget=N); training only, decided 2026-09-30): the actor
cannot hold the fp32 logits of samples much above 16k tokens, and dropping the ~1% longer
episodes left some tasks never trained. With a budget, a tool-phase call whose prompt plus one
tool-use reply (2048) plus a final-answer reserve (FINAL_RESERVE_TOKENS) would pass N goes
straight to final synthesis instead (finalize_reason "training_budget"), and final calls are
capped so the assembled sample fits N. Evaluation passes no budget: rllm's protocol unchanged.

Import after judge_env.load_judge_env(), like finqa_env.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOW_DIR = HERE / "rllm_multitable"
sys.path.insert(0, str(HERE.parent / "finqa_singletable"))

import finqa_env  # noqa: E402  rllm stand-ins, judge client, rllm_finqa on sys.path (finqa_tools, finqa_eval)
from finqa_env import Turn, history_message, tool_call_view  # noqa: E402

for line in (FLOW_DIR / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split()
    actual = hashlib.sha256((FLOW_DIR / name).read_bytes()).hexdigest()
    if actual != digest:
        raise RuntimeError(f"{FLOW_DIR / name}: expected sha256 {digest} (rllm's v2 protocol), got {actual}")

sys.path.insert(0, str(FLOW_DIR))
import multitable_v2_flow as flow  # noqa: E402

from finqa_tools import TOOL_SPECS  # noqa: E402  the flow's own TOOL_SPECS object

if flow.TOOL_SPECS is not TOOL_SPECS:
    raise RuntimeError("multitable_v2_flow imported a different finqa_tools than finqa/rllm_finqa")

MALFORMED_TOOL_CALL_MESSAGE = (
    "Your previous tool call was not valid JSON and could not be executed. "
    "Retry it as one native function call with a JSON object for arguments. "
    "Do not emit literal <tool_call> tags."
)
BUDGET_CHECKPOINT_PREFIX = "Budget checkpoint: "
BUDGET_CHECKPOINT_SUFFIX = (
    " tool-use model turns remain before mandatory final synthesis. "
    "Prioritize unresolved template fields and reserve enough context to produce the complete answer."
)
BUDGET_CHECKPOINTS = {30, 20, 10, 5, 2, 1}
TOOL_USE_CLOSED_MESSAGE = (
    "Tool use is now closed. Using only the evidence already visible in this conversation, produce the "
    "complete final answer now. Begin with `FINAL ANSWER:`. Preserve the requested template, fill every "
    "supported field, include all requested sections, and make no tool calls."
)
EMPTY_FINAL_MESSAGE = "The previous final response was empty. Return the complete `FINAL ANSWER:` text now."
FINAL_RESERVE_TOKENS = 4096  # room kept for the final answer under a training budget
BUDGET_MARGIN_TOKENS = 64


def check_flow_literals() -> None:
    """Every message text above must be a string literal of the vendored flow."""
    tree = ast.parse((FLOW_DIR / "multitable_v2_flow.py").read_text())
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    for text in (MALFORMED_TOOL_CALL_MESSAGE, BUDGET_CHECKPOINT_PREFIX, BUDGET_CHECKPOINT_SUFFIX,
                 TOOL_USE_CLOSED_MESSAGE, EMPTY_FINAL_MESSAGE):
        if text not in literals:
            raise RuntimeError(f"multitable_env message is not a literal of the vendored flow: {text[:60]!r}")
    if "{30, 20, 10, 5, 2, 1}" not in (FLOW_DIR / "multitable_v2_flow.py").read_text():
        raise RuntimeError("the vendored flow's budget checkpoints are no longer {30, 20, 10, 5, 2, 1}")


check_flow_literals()


class ModelCallRejected(RuntimeError):
    """The engine refused a call because of the request itself (a 4xx, a prompt beyond the context).

    The flow's _create_with_retry catches any exception from a call and retries it once; after
    two failures the tool phase moves on to final synthesis and a final attempt counts as spent.
    Infrastructure failures must not be raised as this: they end the episode unscored instead.
    """


class ChatModel(ABC):
    """One model call: the driver's goes through Reef, an evaluator's to vLLM."""

    @abstractmethod
    def complete(self, messages: list[dict], max_tokens: int) -> tuple[dict, dict]:
        """Return (the assistant message as an OpenAI chat dict, per-call info such as the receipt).

        Always sends TOOL_SPECS; raises ModelCallRejected for a request the engine refused.
        """

    def prompt_tokens(self, messages: list[dict]) -> int:
        """Tokens of the next call's prompt (chat template with TOOL_SPECS); needed only under a training budget."""
        raise NotImplementedError("a training budget needs a ChatModel that counts prompt tokens")


@dataclass
class MultiTableEpisode:
    answer: str
    accessed_tables: list[str]
    turns: list[Turn]
    finalize_reason: str  # model_final | tool_turn_budget | context_reserve | tool_phase_llm_error | training_budget
    final_fallback_used: bool
    tool_calls: int = 0
    tool_errors: int = 0
    malformed_tool_calls: int = 0
    llm_errors: list[str] = field(default_factory=list)
    transcript_chars: int = 0


def call_with_retry(chat: ChatModel, messages: list[dict], max_tokens: int) -> tuple[dict | None, dict, list[str]]:
    """flow._create_with_retry: LLM_RETRIES_PER_TURN attempts; None after that many rejections."""
    errors: list[str] = []
    for attempt in range(1, flow.LLM_RETRIES_PER_TURN + 1):
        try:
            message, info = chat.complete(messages, max_tokens)
            return message, info, errors
        except ModelCallRejected as error:
            errors.append(f"attempt={attempt}:{type(error).__name__}:{error}")
    return None, {}, errors


def run_episode(chat: ChatModel, task: dict, token_budget: int | None = None) -> MultiTableEpisode:
    """flow.finqa_multitable_v2 with chat.complete in place of the OpenAI client (+ optional training budget)."""
    messages = flow.build_policy_visible_initial_messages(task, task["question"], str(task["question_id"]))
    accessed: list[str] = []
    turns: list[Turn] = []
    episode = MultiTableEpisode("", accessed, turns, "tool_turn_budget", False)
    final_response = ""
    last_nonempty_content = ""

    for tool_turn in range(flow.MAX_TOOL_TURNS):
        if flow._transcript_chars(messages) >= flow.FINALIZE_AT_TRANSCRIPT_CHARS:
            episode.finalize_reason = "context_reserve"
            break
        if token_budget is not None and (chat.prompt_tokens(messages) + flow.DISCOVERY_MAX_COMPLETION_TOKENS
                                         + FINAL_RESERVE_TOKENS > token_budget):
            episode.finalize_reason = "training_budget"
            break
        message, info, errors = call_with_retry(chat, messages, flow.DISCOVERY_MAX_COMPLETION_TOKENS)
        episode.llm_errors.extend(errors)
        if message is None:
            episode.finalize_reason = "tool_phase_llm_error"
            break
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        messages.append(history_message(message))
        turns.append(Turn(content, len(tool_calls), info))
        if content.strip():
            last_nonempty_content = content
        if not tool_calls and flow._contains_unparsed_tool_call(content):
            episode.malformed_tool_calls += 1
            messages.append({"role": "system", "content": MALFORMED_TOOL_CALL_MESSAGE})
            continue
        if not tool_calls:
            final_response = content
            episode.finalize_reason = "model_final"
            break
        for tc in tool_calls:
            output, had_error = flow._exec_tool_call(tool_call_view(tc), accessed)
            episode.tool_calls += 1
            episode.tool_errors += int(had_error)
            messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": output})
        remaining = flow.MAX_TOOL_TURNS - tool_turn - 1
        if remaining in BUDGET_CHECKPOINTS:
            messages.append({"role": "system",
                             "content": f"{BUDGET_CHECKPOINT_PREFIX}{remaining}{BUDGET_CHECKPOINT_SUFFIX}"})

    if not final_response.strip():
        messages.append({"role": "system", "content": TOOL_USE_CLOSED_MESSAGE})
        for _ in range(flow.FINAL_ATTEMPTS):
            final_max = flow.FINAL_MAX_COMPLETION_TOKENS
            if token_budget is not None:
                final_max = max(256, min(final_max, token_budget - chat.prompt_tokens(messages) - BUDGET_MARGIN_TOKENS))
            message, info, errors = call_with_retry(chat, messages, final_max)
            episode.llm_errors.extend(errors)
            if message is None:
                continue
            content = message.get("content") or ""
            messages.append(history_message(message))
            turns.append(Turn(content, len(message.get("tool_calls") or []), info))
            if content.strip():
                final_response = content
                last_nonempty_content = content
                break
            messages.append({"role": "system", "content": EMPTY_FINAL_MESSAGE})

    if not final_response.strip() and last_nonempty_content.strip():
        final_response = last_nonempty_content
        episode.final_fallback_used = True
    episode.answer = final_response
    episode.transcript_chars = flow._transcript_chars(messages)
    return episode


def grade(task: dict, episode: MultiTableEpisode) -> tuple[float, bool, dict]:
    """rllm finqa_evaluator on the flow's artifacts: (rubric score, score >= 0.9, metadata)."""
    stand_in = sys.modules["rllm.types"].Episode(
        artifacts={"answer": episode.answer, "accessed_tables": episode.accessed_tables, "turns": len(episode.turns)}
    )
    out = finqa_env.finqa_eval.finqa_evaluator(task, stand_in)
    signals = {s.name: s.value for s in out.signals}
    return float(out.reward), bool(out.is_correct), {**out.metadata, **signals}


__all__ = ["TOOL_SPECS", "ChatModel", "ModelCallRejected", "MultiTableEpisode", "run_episode", "grade", "flow"]
