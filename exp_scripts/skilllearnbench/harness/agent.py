"""A minimal bash agent for SkillLearnBench Harbor tasks, served by Reef (training) or vLLM (evaluation).

One Harbor trial is one episode: the task instruction is the first user turn, the model answers with
one ``bash`` tool call per turn, the command runs in the task container (as root, in /root), and its
output comes back as the tool result. The episode ends when the model answers without a tool call,
after ``SLB_MAX_TURNS`` turns, when the next prompt would not fit ``SLB_CONTEXT_TOKENS``, or at
``SLB_EPISODE_S`` seconds outside model calls (a model call may wait through a Reef training step, so
its time does not count). Harbor then runs the task's own verifier.

Training (``SLB_MODE=train``): model calls go through a reef-client proxy on the host's loopback,
which stamps the scenario and token on every call and keeps each call with its Reef receipt (the
ceobench pattern). The receipts and per-call token counts go into the trial's agent metadata, and
run.py reports them after the verifier with the ctrf test-pass fraction as the score.

Evaluation (``SLB_MODE=eval``): the same loop calls ``SLB_BASE_URL`` directly (vLLM serving the base
model or a LoRA adapter); nothing is reported.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer

import httpx
import openai
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client.serve import CaptureStore, ServeConfig, build_handler

SYSTEM_PROMPT = """You are an agent working in a Linux container. You solve the user's task by running shell commands.

- Call the `bash` tool with one command per turn. You see its stdout, stderr and exit code.
- The working directory is /root. Python 3 is available; write helper scripts to files when a command gets long.
- Skills (SKILL.md files with instructions and scripts for this kind of task) are in /root/.agents/skills. Read the relevant SKILL.md before you start.
- Write every output file at the exact path the task names.
- When the task is complete, reply with a short summary and no tool call. That ends the episode."""

#: SLB_PROMPT_VERSION=v2 (user, 2026-10-14): the base model rarely read SKILL.md and ran out of context
#: before it wrote any output file (all tests then fail), so v2 asks for the skill first, an early output
#: file, and short command output. v1 (the default) is unchanged.
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT + """

Work in this order:
1. First list /root/.agents/skills and read the SKILL.md that matches the task. Use its scripts when they fit.
2. Write the required output file early: a first complete version as soon as you can, then improve it. The episode can end before you finish exploring, and a missing output file fails every test.
3. Keep command output short: print only what you need (head, wc, a few rows, summaries). Long output is cut and fills your context."""

#: SLB_PROMPT_VERSION=v3 (user, 2026-10-15): v2 plus a bold, standing reminder about the context budget, so the
#: model plans for it from the first turn instead of only reacting to a budget notice.
#: {context_k} is SLB_CONTEXT_TOKENS / 1024 (32 at the formal 32768), so a longer-context test states its own window.
SYSTEM_PROMPT_V3 = SYSTEM_PROMPT_V2 + """

**Your context window is limited ({context_k}k tokens) and the episode ends without warning when it is full. Track the remaining budget all the time: keep every command's output small, never print whole files or large tables, and keep the required output file written and up to date at every step, so that stopping at any moment still leaves a usable result.**"""

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run one bash command in the task container (cwd /root) and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to run."}},
            "required": ["command"],
        },
    },
}
NO_TOOL_ARGUMENTS = "Error: the bash tool needs a JSON object with a 'command' string."


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, "") or default)


class HarborAgent(BaseAgent):
    """One Harbor trial: the bash loop above, one tool call per turn."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        environ = os.environ
        self.mode = environ.get("SLB_MODE", "train")
        if self.mode not in ("train", "eval"):
            raise ValueError(f"SLB_MODE must be train or eval, got {self.mode!r}")
        self.service_url = environ.get("REEF_SERVICE_URL", "").rstrip("/")
        self.scenario = environ.get("REEF_SCENARIO", "slb-sao")
        self.token = environ.get("REEF_TOKEN", "reef-local")
        self.base_url = environ.get("SLB_BASE_URL", "").rstrip("/")
        if self.mode == "train" and not self.service_url:
            raise ValueError("SLB_MODE=train needs REEF_SERVICE_URL")
        if self.mode == "eval" and not self.base_url:
            raise ValueError("SLB_MODE=eval needs SLB_BASE_URL (an OpenAI-compatible /v1 URL)")
        self.temperature = env_float("SLB_TEMPERATURE", 1.0)
        self.max_turns = int(env_float("SLB_MAX_TURNS", 60))
        self.max_completion_tokens = int(env_float("SLB_MAX_COMPLETION_TOKENS", 4096))
        self.context_tokens = int(env_float("SLB_CONTEXT_TOKENS", 32768))
        self.episode_s = env_float("SLB_EPISODE_S", 1800)
        self.command_timeout_s = int(env_float("SLB_COMMAND_TIMEOUT_S", 600))
        self.output_chars = int(env_float("SLB_OUTPUT_CHARS", 8000))
        self.llm_timeout_s = env_float("SLB_LLM_TIMEOUT_S", 7200)
        # Qwen3.5 thinking mode (chat_template_kwargs.enable_thinking); off for training (user, 2026-10-14).
        self.enable_thinking = os.environ.get("SLB_ENABLE_THINKING", "0") == "1"
        prompt_version = os.environ.get("SLB_PROMPT_VERSION", "v1")
        prompts = {"v1": SYSTEM_PROMPT, "v2": SYSTEM_PROMPT_V2, "v3": SYSTEM_PROMPT_V3}
        if prompt_version not in prompts:
            raise ValueError(f"SLB_PROMPT_VERSION must be one of {sorted(prompts)}, got {prompt_version!r}")
        self.system_prompt = prompts[prompt_version].replace("{context_k}", str(round(self.context_tokens / 1024)))
        # SLB_BUDGET_WARN (user, 2026-10-15): context fractions (e.g. "0.6,0.8") at which a budget notice is appended
        # as a user turn after the tool results, so the model writes its output before the context runs out. Off
        # when empty (the default). Append-only, so Reef can still assemble the episode into one sample.
        self.budget_warn = sorted(float(x) for x in os.environ.get("SLB_BUDGET_WARN", "").split(",") if x.strip())
        # SLB_BUDGET_NOTICE_IN (2026-10-20): where the notice goes. "tool" (default) appends it to the turn's last tool
        # result. "user" sends it as its own user turn, as the 2026-10-15/19 evals did; Qwen3.5's chat template then
        # drops the empty <think></think> of every earlier assistant turn (they now precede the last user query), the
        # history re-renders differently, and Reef cannot assemble the episode (full20 smoke: 12/12 unassemblable).
        self.budget_notice_in = os.environ.get("SLB_BUDGET_NOTICE_IN", "tool")
        if self.budget_notice_in not in ("tool", "user"):
            raise ValueError(f"SLB_BUDGET_NOTICE_IN must be tool or user, got {self.budget_notice_in!r}")
        self.capture = CaptureStore()

    @staticmethod
    def name() -> str:
        return "slb-bash"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the task image has what the task needs."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        loop = asyncio.get_running_loop()
        proxy = self.start_proxy() if self.mode == "train" else None
        base_url = f"http://127.0.0.1:{proxy.server_address[1]}/v1" if proxy else self.base_url
        try:
            outcome = await asyncio.to_thread(self.play, instruction, environment, loop, base_url)
        finally:
            if proxy is not None:
                proxy.shutdown()
        turns = [t for t in self.capture.snapshot() if t.get("status") == 200 and t.get("receipt")]
        usage = [((t.get("response") or {}).get("usage") or {}) for t in turns]
        context.metadata = {
            **(context.metadata or {}),
            "reef": {
                "agent_record_ids": [t["receipt"] for t in turns],
                "agent_record_tokens": [
                    int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0) for u in usage
                ],
            },
            "slb": outcome,
        }
        context.n_input_tokens = outcome["prompt_tokens"]
        context.n_output_tokens = outcome["completion_tokens"]

    def start_proxy(self) -> ThreadingHTTPServer:
        """The ceobench proxy: the agent's OpenAI client talks to it, Reef gets the calls and the receipts."""
        config = ServeConfig(
            upstream=self.service_url,
            listen_host="127.0.0.1",
            listen_port=0,
            override_headers={"x-reef-scenario": self.scenario, "authorization": f"Bearer {self.token}"},
        )
        self.capture = CaptureStore()
        server = ThreadingHTTPServer((config.listen_host, config.listen_port), build_handler(config, self.capture))
        threading.Thread(target=server.serve_forever, name="slb-proxy", daemon=True).start()
        return server

    def bash(self, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop, command: str) -> str:
        """Run one command in the task container and format its output like a terminal transcript."""
        future = asyncio.run_coroutine_threadsafe(
            environment.exec(command, cwd="/root", timeout_sec=self.command_timeout_s), loop
        )
        try:
            result = future.result(timeout=self.command_timeout_s + 60)
        except Exception as error:  # noqa: BLE001 - a failed command is the model's problem, not the trial's
            return f"Error: the command failed to run: {type(error).__name__}: {error}"
        parts = [result.stdout or ""]
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        parts.append(f"[exit code: {result.return_code}]")
        output = "\n".join(part for part in parts if part)
        if len(output) > self.output_chars:
            half = self.output_chars // 2
            output = output[:half] + f"\n... ({len(output) - self.output_chars} characters cut) ...\n" + output[-half:]
        return output

    def play(self, instruction: str, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop, base_url: str) -> dict:
        model = os.environ.get("SLB_MODEL", "") or self.model_name or "reef"
        client = openai.OpenAI(base_url=base_url, api_key="reef", timeout=httpx.Timeout(self.llm_timeout_s))
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": instruction}]
        started = time.time()
        model_s = 0.0  # time inside model calls: it includes Reef's training pauses, so the limit excludes it
        prompt_tokens = completion_tokens = turns = 0
        warned: set = set()  # budget notices already sent (context fractions, or "turns")
        end = "max_turns"
        with open(self.logs_dir / "turns.jsonl", "a", encoding="utf-8") as log:
            while turns < self.max_turns:
                if time.time() - started - model_s > self.episode_s:
                    end = "time_limit"
                    break
                call_started = time.time()
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=[BASH_TOOL],
                        tool_choice="auto",
                        temperature=self.temperature,
                        max_tokens=self.max_completion_tokens,
                        extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
                    )
                except openai.BadRequestError as error:  # e.g. the prompt no longer fits the context
                    end = f"bad_request: {str(error)[:200]}"
                    break
                finally:
                    model_s += time.time() - call_started
                turns += 1
                usage = response.usage
                prompt_tokens += usage.prompt_tokens if usage else 0
                completion_tokens += usage.completion_tokens if usage else 0
                message = response.choices[0].message
                calls = list(message.tool_calls or [])
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        **(
                            {
                                "tool_calls": [
                                    {
                                        "id": c.id,
                                        "type": "function",
                                        "function": {"name": c.function.name, "arguments": c.function.arguments},
                                    }
                                    for c in calls
                                ]
                            }
                            if calls
                            else {}
                        ),
                    }
                )
                if not calls:
                    end = "answered"
                    log.write(json.dumps({"turn": turns, "end": end, "content": message.content}) + "\n")
                    break
                for index, call in enumerate(calls):
                    if index > 0:
                        result = "[Skipped - only one tool call per turn.]"
                    else:
                        try:
                            command = json.loads(call.function.arguments or "{}").get("command", "")
                        except (json.JSONDecodeError, AttributeError):
                            command = ""
                        result = self.bash(environment, loop, command) if command else NO_TOOL_ARGUMENTS
                        log.write(json.dumps({"turn": turns, "command": command, "output": result[:2000]}) + "\n")
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                log.flush()
                used = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
                if used + self.max_completion_tokens + 2048 > self.context_tokens:
                    end = "context_full"
                    break
                due = [f for f in self.budget_warn if used >= f * self.context_tokens and f not in warned]
                turns_left = self.max_turns - turns
                if due or (self.budget_warn and turns_left == 5 and "turns" not in warned):
                    warned.update(due or ["turns"])
                    left = max(0, self.context_tokens - used - self.max_completion_tokens - 2048)
                    notice = (
                        f"[Budget notice] About {left} tokens of context and {turns_left} turns are left before this "
                        "episode ends. If the required output file is not written yet, write a complete version now. "
                        "Then finish: check the file once and reply with a short summary and no tool call."
                    )
                    if self.budget_notice_in == "tool":
                        messages[-1]["content"] += "\n\n" + notice
                    else:
                        messages.append({"role": "user", "content": notice})
                    log.write(json.dumps({"turn": turns, "budget_notice": notice}) + "\n")
        return {
            "end": end,
            "turns": turns,
            "seconds": round(time.time() - started, 1),
            "model_seconds": round(model_s, 1),
            "budget_notices": len(warned),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
