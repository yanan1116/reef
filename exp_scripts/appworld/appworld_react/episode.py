"""One AppWorld episode under the react_code protocol (dynamic API discovery), stack-agnostic.

This is the AppWorld environment shared by SAO (reef exp_scripts/appworld) and PRPO (rllm
exp_scripts/appworld-prpo-run): the same file, byte for byte, in both repositories, so the
two algorithms see one protocol. It reproduces the grpo_vanilla react_code line (the one
behind the vanilla-GRPO AppWorld results) as its trajectory collector runs it
(grpo_vanilla/scripts/collect_appworld_trajectories.py:run_task with --align-harness,
modality "code"); see SOURCE.txt for commits and hashes, and test_parity.py for the
byte-for-byte checks against that code.

Protocol, as in grpo_vanilla:
  * prompt: system = grpo_vanilla SYSTEM_PROMPT; user = the task instruction followed by
    AppWorld's official react_code few-shot walkthrough (react_code_instructions.txt,
    placeholders filled) and the harness's three closing lines (date/time, the
    execute_code convention, the call budget)
  * one tool, execute_code, whose code runs in the task's AppWorld session
  * MAX_STEPS = 50 executed tool calls, enforced per call: a call past the budget is
    answered with a refusal and not executed
  * EPISODE_TOKEN_WALL = 24576 generated tokens for the whole episode: each call may
    generate at most what is left, and an episode that runs out is truncated
  * tool results clipped at 60000 characters with a trailing "[truncated]" line
  * the episode ends when the model replies without a tool call, when the budget is
    exhausted, or on truncation
  * reward: grpo_vanilla's corrected score, the fraction of the task's no_op_fail tests
    passed (appworld_verifier.grade_rollout); official TGC (all tests) recorded beside it

What a stack supplies: a ChatModel that makes one model call, with the messages, the tool
spec and a max_tokens, and returns the assistant message plus usage. How it samples
(temperature, thinking off) is the stack's business, as in grpo_vanilla it was the caller's.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .grpo_vanilla.appworld_verifier import grade_rollout

HERE = Path(__file__).resolve().parent
REACT_PROMPT_PATH = HERE / "react_code_instructions.txt"

MAX_STEPS = 50                   # AppWorld's official budget for its ReAct/code agents
EPISODE_TOKEN_WALL = 24576       # grpo_vanilla react_code line: max_completion_length
TOOL_RESULT_CHARS = 60_000       # grpo_vanilla config.limits.max_tool_result_chars
INITIALIZE_RANDOM_SEED = 100     # grpo_vanilla _INITIALIZE_RANDOM_SEED (AppWorld's own default)
INITIALIZE_TIMEOUT_S = 600       # collector: /initialize timeout_seconds

# grpo_vanilla/grpo_vanilla/config.py SYSTEM_PROMPT, verbatim (sha256 in SOURCE.txt).
SYSTEM_PROMPT = (
    "You are an autonomous assistant completing an enterprise workflow task.\n"
    "\n"
    "Operating rules:\n"
    "- Use the provided tools to gather every fact you need. Never invent data that a tool "
    "or an attached document did not give you.\n"
    "- Issue tool calls only through the native tool-call format. Never write a tool call as "
    "plain text.\n"
    "- Each tool result is returned to you before you continue, so call a tool, read its "
    "result, then decide the next step.\n"
    "- Work within the stated turn budget. When you have what you need, stop calling tools.\n"
    "- Your final message must be the complete deliverable the task asks for, written out in "
    "full. Do not answer with a plan, a summary of what you would do, or code that would "
    "produce the answer."
)

# grpo_vanilla/grpo_vanilla/appworld_environment.py EXECUTE_CODE_DESCRIPTION / _SCHEMA, verbatim.
EXECUTE_CODE_DESCRIPTION = (
    "Execute Python code in the task's persistent IPython session and return whatever it "
    "prints. The session exposes `apis`, through which every application is reached: call "
    "`apis.api_docs.show_app_descriptions()` to see the applications, "
    "`apis.api_docs.show_api_descriptions(app_name=...)` to list one application's APIs, and "
    "`apis.api_docs.show_api_doc(app_name=..., api_name=...)` for an API's arguments. "
    "Nothing is returned unless the code prints it. Variables persist across calls. "
    "When the task is done call `apis.supervisor.complete_task(...)`, passing `answer=` if "
    "the task asks for one."
)

EXECUTE_CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Python code to execute in the task session.",
        }
    },
    "required": ["code"],
}

TOOLS = [{"type": "function", "function": {
    "name": "execute_code", "description": EXECUTE_CODE_DESCRIPTION, "parameters": dict(EXECUTE_CODE_SCHEMA)}}]


def render_react_prompt(template: str, info: dict[str, Any], budget: int) -> str:
    """grpo_vanilla appworld_environment._render_react_prompt, verbatim."""
    supervisor = info["supervisor"]
    filled = template
    for key, value in (
        ("{{ instruction }}", info["instruction"]),
        ("{{ main_user.first_name }}", supervisor["first_name"]),
        ("{{ main_user.last_name }}", supervisor["last_name"]),
        ("{{ main_user.email }}", supervisor["email"]),
        ("{{ main_user.phone_number }}", supervisor["phone_number"]),
    ):
        filled = filled.replace(key, str(value))
    return (
        filled.rstrip()
        + f"\n\nThe current date and time is {info['datetime']}.\n"
        + "In this environment you do not write fenced code blocks: put the Python into the "
        + "`code` argument of the `execute_code` tool, one step per call, and read the "
        + "printed output from the tool result.\n"
        + f"You have at most {budget} tool calls."
    )


def initial_messages(info: dict[str, Any], template: str, budget: int = MAX_STEPS) -> list[dict]:
    """The collector's --align-harness prompt: SYSTEM_PROMPT, then instruction + harness observation
    (whose react_code branch is "\\n\\n" + the rendered official prompt)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": info["instruction"] + "\n\n" + render_react_prompt(template, info, budget)},
    ]


class AppWorldServer:
    """One `appworld serve environment` process on its own port (one world per process)."""

    def __init__(self, port: int, root: Path, venv: Path, log: Path) -> None:
        self.port, self.root, self.venv, self.log = port, root, venv, log
        self.proc: subprocess.Popen | None = None

    def start(self, ready_timeout_s: float = 240.0) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["APPWORLD_ROOT"] = str(self.root)
        env["APPWORLD_CACHE"] = str(self.root / ".cache")
        env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1"
        self.proc = subprocess.Popen(
            [str(self.venv / "bin" / "appworld"), "serve", "environment",
             "--port", str(self.port), "--root", str(self.root), "--no-show-usage"],
            stdout=self.log.open("ab"), stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        deadline = time.monotonic() + ready_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"appworld server on port {self.port} exited: {self.log.read_text()[-1500:]}")
            with socket.socket() as probe:
                probe.settimeout(1.0)
                if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                    return
            time.sleep(1.0)
        raise RuntimeError(f"appworld server on port {self.port} never became reachable")

    def post(self, route: str, payload: dict, timeout: float = 300.0) -> Any:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{route}", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())["output"]  # serve-environment wraps every payload

    def stop(self) -> None:
        """Kill the server's whole session: it runs in its own, so a caller's process-group kill misses it."""
        if self.proc and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), 15)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), 9)


class ChatModel(ABC):
    """One model call. The stack decides sampling; the episode decides messages, tools and max_tokens."""

    @abstractmethod
    def complete(self, messages: list[dict], tools: list[dict], max_tokens: int) -> tuple[dict, dict]:
        """Return (assistant message dict with optional tool_calls, info). info must hold
        completion_tokens and finish_reason; anything else (receipt, versions) is kept per turn.
        Raise ContextOverflow when the request exceeds the model's context window."""


class ContextOverflow(RuntimeError):
    """The conversation outgrew the served context window: the model's doing, like a truncation."""


@dataclass
class Episode:
    task_id: str
    messages: list[dict]
    turns: list[dict] = field(default_factory=list)         # per model call: info from ChatModel.complete
    dispatches: list[dict] = field(default_factory=list)    # per executed tool call
    stop_reason: str = "budget_exhausted"
    completion_tokens: int = 0
    task_completed: bool = False
    tracker: Any = None
    reward: float | None = None                              # corrected score; None = grader_failure
    tgc: bool = False
    verifier_status: str = ""
    verifier_components: dict = field(default_factory=dict)
    initial_state_sha256: str | None = None
    elapsed_s: float = 0.0


def state_fingerprint(root: Path, experiment: str, task_id: str) -> str | None:
    """sha256 over the databases /initialize seeded (the collector's check that no state leaked)."""
    dbs = root / "experiments" / "outputs" / experiment / "tasks" / task_id / "dbs"
    files = sorted(dbs.glob("*.jsonl")) if dbs.is_dir() else []
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_episode(chat: ChatModel, server: AppWorldServer, task_id: str, experiment: str,
                template: str | None = None, max_steps: int = MAX_STEPS,
                token_wall: int = EPISODE_TOKEN_WALL) -> Episode:
    """The collector's run_task for code modality with --align-harness, turn for turn."""
    template = template if template is not None else REACT_PROMPT_PATH.read_text()
    info = server.post("/initialize", {
        "task_id": task_id, "experiment_name": experiment,
        "max_interactions": max_steps, "timeout_seconds": INITIALIZE_TIMEOUT_S,
        "load_ground_truth": False, "random_seed": INITIALIZE_RANDOM_SEED,
        "include_direct_functions": False})
    episode = Episode(task_id=task_id, messages=initial_messages(info, template, max_steps))
    episode.initial_state_sha256 = state_fingerprint(server.root, experiment, task_id)
    tokens_left = token_wall
    started = time.monotonic()
    for _ in range(max_steps + 4):
        if tokens_left <= 0:
            episode.stop_reason = "truncated"
            break
        try:
            reply, turn = chat.complete(episode.messages, TOOLS, tokens_left)
        except ContextOverflow as error:
            episode.stop_reason = "context_overflow"
            episode.turns.append({"failure": str(error)[:500]})
            break
        turn_tokens = int(turn.get("completion_tokens") or 0)
        episode.completion_tokens += turn_tokens
        tokens_left -= turn_tokens
        episode.turns.append(turn)
        calls = reply.get("tool_calls") or []
        episode.messages.append({"role": "assistant", "content": reply.get("content") or "",
                                 **({"tool_calls": calls} if calls else {})})
        if not calls:
            episode.stop_reason = "truncated" if (turn.get("finish_reason") == "length" or tokens_left <= 0) else "no_tool_call"
            break
        if len(episode.dispatches) >= max_steps:
            episode.stop_reason = "budget_exhausted"
            break
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            raw = function.get("arguments")
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            except json.JSONDecodeError:
                args = {}
            if len(episode.dispatches) >= max_steps:
                episode.messages.append({"role": "tool", "name": name, "tool_call_id": call.get("id") or name,
                                         "content": json.dumps({"error": "interaction budget exhausted for this task",
                                                                "budget": max_steps})})
                episode.stop_reason = "budget_exhausted"
                continue
            code = args.get("code", "") if (name == "execute_code" and isinstance(args, dict)) else f"print({name}(**{args!r}))"
            try:
                output = server.post("/execute", {"task_id": task_id, "code": code})
                text = output if isinstance(output, str) else json.dumps(output)
                outcome = "ok"
            except Exception as error:  # noqa: BLE001 - the policy sees the error, as in the collector
                text, outcome = json.dumps({"error": repr(error)}), "error"
            episode.dispatches.append({"tool_name": name, "code": code, "result_chars": len(text), "outcome": outcome})
            shown = text[:TOOL_RESULT_CHARS] + "\n[truncated]" if len(text) > TOOL_RESULT_CHARS else text
            episode.messages.append({"role": "tool", "name": name, "tool_call_id": call.get("id") or name,
                                     "content": shown})
    episode.task_completed = bool(server.post("/task_completed", {"task_id": task_id}))
    episode.tracker = server.post("/evaluate", {"task_id": task_id, "suppress_errors": True,
                                                "experiment_name": experiment})
    outcome = grade_rollout(task_id=task_id, audit_data={"appworld": {
        "tracker": episode.tracker, "experiment_name": experiment,
        "task_completed": episode.task_completed, "interactions": len(episode.dispatches)}})
    episode.reward, episode.tgc = outcome.score, bool(outcome.passed)
    episode.verifier_status, episode.verifier_components = outcome.status, dict(outcome.components or {})
    episode.elapsed_s = round(time.monotonic() - started, 2)
    return episode


__all__ = ["MAX_STEPS", "EPISODE_TOKEN_WALL", "SYSTEM_PROMPT", "TOOLS", "AppWorldServer", "ChatModel",
           "ContextOverflow", "Episode", "initial_messages", "render_react_prompt", "run_episode"]
