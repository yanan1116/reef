"""The FinQA environment PRPO trained and was evaluated on, without rllm.

rllm_finqa/ holds byte-identical copies of rllm's cookbooks/finqa files (see
rllm_finqa/SOURCE.txt). This module loads them and exposes the two things both the
SAO training driver and the evaluator need:

  run_episode(chat, question)  the multi-turn ReAct loop of rllm's finqa_flow,
                               step for step, with the model call injected: the
                               driver routes it through Reef (one receipt per
                               turn), the evaluator calls vLLM directly.
  grade(task, episode)         rllm's finqa_evaluator (judge-LLM correctness;
                               the reward is correctness alone, the table-access
                               score is a signal only).

rllm itself is never imported. The copied files reference three rllm names at
import time (a decorator and a few types); they get minimal in-memory stand-ins
here. A real rllm already in sys.modules is refused rather than mixed in.

Environment (read at import):
  FINQA_TABLES_ROOT   company tables; default exp_scripts/finqa/data/company_tables
  OPENAI_API_KEY, OPENAI_BASE_URL, FINQA_JUDGE_MODEL, FINQA_JUDGE_MAX_ATTEMPTS,
  FINQA_JUDGE_FINISH_LOG  the judge, exactly as in the PRPO runs; call
                          judge_env.load_judge_env() before importing this module
"""

from __future__ import annotations

import os
import sys
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "rllm_finqa"  # not "env/": the repository .gitignore ignores env/ directories
DATA_DIR = Path(os.environ.get("FINQA_DATA_DIR", str(HERE / "data")))


class RllmStandInModule(types.ModuleType):
    """Marks the in-memory modules below, so a real rllm in sys.modules can be told apart."""


def install_rllm_stand_ins() -> None:
    """Satisfy the copied files' rllm imports with inert stand-ins (no rllm code runs)."""
    existing = sys.modules.get("rllm")
    if existing is not None and not isinstance(existing, RllmStandInModule):
        raise RuntimeError("the real rllm is imported in this process; the SAO FinQA env must not mix with it")

    @dataclass
    class Signal:
        name: str
        value: float

    @dataclass
    class EvalOutput:
        reward: float
        is_correct: bool
        signals: list = field(default_factory=list)
        metadata: dict = field(default_factory=dict)

    class Task:  # finqa_eval only uses it in an isinstance check; tasks here are plain dicts
        metadata: dict | None = None

    @dataclass
    class Episode:
        trajectories: list = field(default_factory=list)
        artifacts: dict = field(default_factory=dict)

    @dataclass
    class Step:
        chat_completions: list = field(default_factory=list)
        model_response: str = ""
        action: str = ""
        thought: str = ""

    @dataclass
    class Trajectory:
        name: str = ""
        steps: list = field(default_factory=list)

    @dataclass
    class AgentConfig:
        base_url: str = ""
        model: str = ""

    def evaluator(fn):
        return fn

    def rollout(**options):
        def register(fn):
            return fn

        return register

    rllm = RllmStandInModule("rllm")
    rllm.evaluator, rllm.rollout = evaluator, rollout
    rllm_types = RllmStandInModule("rllm.types")
    rllm_types.Task, rllm_types.Episode, rllm_types.Step = Task, Episode, Step
    rllm_types.Trajectory, rllm_types.AgentConfig = Trajectory, AgentConfig
    rllm_eval = RllmStandInModule("rllm.eval")
    rllm_eval_types = RllmStandInModule("rllm.eval.types")
    rllm_eval_types.EvalOutput, rllm_eval_types.Signal = EvalOutput, Signal
    rllm.types, rllm.eval, rllm_eval.types = rllm_types, rllm_eval, rllm_eval_types
    sys.modules.update({"rllm": rllm, "rllm.types": rllm_types, "rllm.eval": rllm_eval, "rllm.eval.types": rllm_eval_types})


os.environ.setdefault("FINQA_TABLES_ROOT", str(DATA_DIR / "company_tables"))
if not os.path.isdir(os.environ["FINQA_TABLES_ROOT"]):
    raise RuntimeError(
        f"FinQA tables not found at {os.environ['FINQA_TABLES_ROOT']}: copy the PRPO data into "
        f"{DATA_DIR} (check it against data_manifest.sha256) or set FINQA_TABLES_ROOT"
    )
if not os.environ.get("OPENAI_API_KEY"):
    # finqa_eval builds its judge client at import and silently scores every answer 0 without a key.
    raise RuntimeError("OPENAI_API_KEY is unset: call judge_env.load_judge_env() before importing finqa_env")

install_rllm_stand_ins()
sys.path.insert(0, str(SOURCE_DIR))
import finqa_eval  # noqa: E402  (copied file; builds the judge client from OPENAI_API_KEY)
import finqa_flow  # noqa: E402  (copied file; only its pure helpers are used)
from finqa_tools import TOOL_SPECS  # noqa: E402  (copied file; preloads every company table into sqlite)

SYSTEM_PROMPT: str = finqa_flow.SYSTEM_PROMPT
MAX_TURNS: int = finqa_flow.MAX_TURNS
if MAX_TURNS != 20:
    raise RuntimeError(f"finqa_flow.MAX_TURNS is {MAX_TURNS}; the PRPO runs used 20")


@dataclass
class Turn:
    """One model call of an episode."""

    content: str
    tool_calls: int
    info: dict  # whatever the chat callable returned beside the message (receipt, usage, ...)


@dataclass
class EpisodeResult:
    answer: str
    accessed_tables: list[str]
    turns: list[Turn]
    ended: str  # "answer" | "max_turns" | "call_failed" | "tool_error"
    failure: Exception | None = None  # the exception that ended a "call_failed" or "tool_error" episode


class ChatModel(ABC):
    """One model call of an episode: the driver's goes through Reef, the evaluator's to vLLM."""

    @abstractmethod
    def complete(self, messages: list[dict]) -> tuple[dict, dict]:
        """Return (the assistant message as an OpenAI chat dict, per-call info such as the receipt)."""


def tool_call_view(tc: dict) -> types.SimpleNamespace:
    """The attribute view finqa_flow._exec_tool_call reads (it was written for OpenAI SDK objects)."""
    fn = tc.get("function") or {}
    return types.SimpleNamespace(
        id=tc.get("id"),
        type=tc.get("type", "function"),
        function=types.SimpleNamespace(name=fn.get("name"), arguments=fn.get("arguments")),
    )


def history_message(message: dict) -> dict:
    """finqa_flow._msg_to_dict applied to an SDK message: role, content only if non-empty, clean tool_calls."""
    out: dict = {"role": message.get("role") or "assistant"}
    if message.get("content"):
        out["content"] = message["content"]
    if message.get("tool_calls"):
        out["tool_calls"] = [
            {
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": {"name": (tc.get("function") or {}).get("name"), "arguments": (tc.get("function") or {}).get("arguments")},
            }
            for tc in message["tool_calls"]
        ]
    return out


def run_episode(chat: ChatModel, question: str) -> EpisodeResult:
    """rllm finqa_flow's loop, turn for turn.

    chat.complete(messages) returns (assistant message dict, info) and owns the
    tool specs and sampling; an exception ends the episode the way a failed LLM
    call ends finqa_flow: no answer, so the judge scores it 0.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    accessed: list[str] = []
    turns: list[Turn] = []
    for _ in range(MAX_TURNS):
        try:
            message, info = chat.complete(messages)
        except Exception as error:  # finqa_flow: log and break with final_response == ""
            return EpisodeResult("", accessed, turns, "call_failed", error)
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        messages.append(history_message(message))
        turns.append(Turn(content, len(tool_calls), info))
        if not tool_calls:
            return EpisodeResult(content, accessed, turns, "answer")
        for tc in tool_calls:
            try:
                output = finqa_flow._exec_tool_call(tool_call_view(tc), accessed)
            except Exception as error:
                # finqa_flow lets this escape (e.g. arguments that decode to a JSON string: args.get on a
                # str, seen 2026-09-27 00:3x in the formal run); rllm's eval runner then records the episode
                # as TerminationReason.ERROR with reward 0 and is_correct False. Same here: no answer.
                return EpisodeResult("", accessed, turns, "tool_error", error)
            messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": output})
    return EpisodeResult(turns[-1].content if turns else "", accessed, turns, "max_turns")


def grade(task: dict, episode: EpisodeResult) -> tuple[float, bool, dict]:
    """rllm finqa_evaluator: (reward, is_correct, metadata incl. table_access)."""
    stand_in = sys.modules["rllm.types"].Episode(
        artifacts={"answer": episode.answer, "accessed_tables": episode.accessed_tables, "turns": len(episode.turns)}
    )
    out = finqa_eval.finqa_evaluator(task, stand_in)
    signals = {s.name: s.value for s in out.signals}
    return float(out.reward), bool(out.is_correct), {**out.metadata, **signals}


__all__ = ["SYSTEM_PROMPT", "MAX_TURNS", "TOOL_SPECS", "ChatModel", "run_episode", "grade", "EpisodeResult", "Turn"]
