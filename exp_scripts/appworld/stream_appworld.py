"""Stream AppWorld episodes through Reef for SAO, N in flight, one sample per episode.

The FinQA driver (finqa/stream_finqa.py) with the benchmark swapped: SAO, the recipe
(finqa/sao_multiturn.py MultiTurnSAORecipe), pacing, budget, the pre-report assembly check
and the failure policy are the same. What differs is only the episode:

* A rollout is one AppWorld episode under the react_code protocol (appworld_react/episode.py,
  byte-identical to the rllm PRPO copy): the official few-shot prompt, one execute_code tool,
  50 executed calls, 24576 generated tokens for the whole episode, every model call sent
  through Reef's /v1/chat/completions for its receipt. Each episode gets its own
  `appworld serve environment` process (one world per process) on a port from a pool; the
  world it wrote under APPWORLD_ROOT/experiments/outputs/ is deleted once graded.
* Sampling: temperature 1.0, top_p 1.0, as the grpo_vanilla AppWorld line (the recipe's
  rollout-temperature must match: SAO's DIS ratio compares against these log-probs).
  Qwen3-4B-Instruct-2507 has no thinking mode.
* Context: the served context is CONTEXT_TOKENS. The protocol caps only the episode's
  generated tokens, so each call's max_tokens is the smaller of what is left of the episode
  wall and what is left of the context after this prompt, counted with the model's chat
  template; a prompt that no longer fits ends the episode as context_overflow (graded as
  it stands, like a truncation).
* Score: grpo_vanilla's corrected reward (fraction of the task's no_op_fail tests passed),
  deterministic, no judge. A task AppWorld cannot grade (score None) is dropped and not retried.
* Before reporting, every turn's Reef record is read back and Reef's own
  make_multi_turn_policy_trajectory is run on them; an episode Reef cannot assemble is
  recorded as dropped and its task goes back into the queue (as in the FinQA driver).

Environment (defaults are the AppWorld formal run):
  SAO_PROBLEMS        task-id list, one per line (default: AppWorld data/datasets/train.txt, 90 tasks)
  SAO_SCENARIO        Reef scenario (default sao-appworld)
  SAO_BATCH           episodes per optimizer step; = recipe batch-size (default 30)
  SAO_IN_FLIGHT       concurrent episodes (default 30)
  SAO_BUDGET          reported episodes before the driver stops (default 2700 = 90 steps x 30 = 30 epochs)
  SAO_TEMPERATURE, SAO_TOP_P   sampling (1.0, 1.0)
  SAO_CONTEXT_TOKENS  served context window (default 32768 = the recipe's context-length)
  SAO_MAX_SAMPLE_TOKENS  longest assembled sample reported for training (default 16384). A longer
                      episode is dropped "too_long" and not re-queued: a sample-selection rule (the
                      run is length-filtered SAO; report its filter rate), forced because fp32
                      [tokens, vocab] logits of one ~31k-token sample need 18.6 GiB (multi-table
                      smoke, 2026-09-29) and the bf16-logits patch trips Slime's fp32 assertion.
  SAO_MODEL_PATH      tokenizer used to count prompt tokens (default the 2507 HF snapshot)
  APPWORLD_ROOT, APPWORLD_VENV, APPWORLD_PORT_BASE   AppWorld checkout, its venv, first server port (7800)
  SAO_RECORDS_PATH, SAO_EPISODES_PATH   one JSON line per episode: summary / full conversation
  SAO_PROGRESS_FILE, SAO_AHEAD, SAO_STALL_S, SAO_SEED, SAO_TRAIN_DRAIN_TIMEOUT_S  as in the FinQA driver
"""

from __future__ import annotations

import json
import os
import queue as queue_module
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from reef_client import ReefClient, ReefClientError

HERE = Path(__file__).resolve().parent
REEF_ROOT = HERE.parents[1]  # the reef fork checkout the image was built from
sys.path.insert(0, str(HERE))
# Reef's assembly code, imported from the checkout (as the FinQA driver): tomli_w lives in exp_scripts/.reef-deps.
sys.path[1:1] = [str(REEF_ROOT), str(HERE.parent / ".reef-deps")]
from reef.core.records_types import AgentRecord, RequestType  # noqa: E402
from reef.train.processors.common import make_multi_turn_policy_trajectory  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from appworld_react.episode import (  # noqa: E402
    AppWorldServer, ChatModel, ContextOverflow, Episode, run_episode,
)

SERVICE_URL = "http://127.0.0.1:8900"
TOKEN = "reef-local"
SCENARIO = os.environ.get("SAO_SCENARIO", "sao-appworld")
RECIPE = "sao"

TAIL = Path("/home/yanan/agents/gitlab/tail")
APPWORLD_ROOT = Path(os.environ.get("APPWORLD_ROOT", str(TAIL / "appworld")))
APPWORLD_VENV = Path(os.environ.get("APPWORLD_VENV", str(APPWORLD_ROOT / ".venv")))
APPWORLD_PORT_BASE = int(os.environ.get("APPWORLD_PORT_BASE", "7800"))
BATCH = int(os.environ.get("SAO_BATCH", "30"))
IN_FLIGHT = int(os.environ.get("SAO_IN_FLIGHT", "30"))
BUDGET = int(os.environ.get("SAO_BUDGET", str(90 * 30)))
TEMPERATURE = float(os.environ.get("SAO_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("SAO_TOP_P", "1.0"))
CONTEXT_TOKENS = int(os.environ.get("SAO_CONTEXT_TOKENS", "32768"))
MAX_SAMPLE_TOKENS = int(os.environ.get("SAO_MAX_SAMPLE_TOKENS", "16384"))
CONTEXT_MARGIN = 64  # tolerance between the local chat-template count and the engine's
MODEL_PATH = os.environ.get(
    "SAO_MODEL_PATH",
    "/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
)
CALL_TIMEOUT_S = float(os.environ.get("SAO_CALL_TIMEOUT_S", "1800"))
RECORDS_PATH = Path(os.environ.get("SAO_RECORDS_PATH", "work/records/appworld.jsonl"))
EPISODES_PATH = Path(os.environ.get("SAO_EPISODES_PATH", str(RECORDS_PATH.with_name("episodes.jsonl"))))
SEED = int(os.environ.get("SAO_SEED", "0"))
TRAIN_DRAIN_TIMEOUT_S = int(os.environ.get("SAO_TRAIN_DRAIN_TIMEOUT_S", "14400"))
PROGRESS_FILE = os.environ.get("SAO_PROGRESS_FILE")
AHEAD = int(os.environ.get("SAO_AHEAD", "3"))
STALL_S = int(os.environ.get("SAO_STALL_S", "2700"))
RUN_TAG = os.environ.get("SAO_RUN_TAG", time.strftime("%Y%m%dT%H%M%S"))
REALIGN_THRESHOLD = 1024  # = sao_multiturn.MultiTurnSAORecipe realign_threshold
SCAFFOLD_TOLERANCE = 0    # = sao_multiturn.MultiTurnSAORecipe scaffold_tolerance
MAX_FAILURE_STREAK = 24
FAILURE_PAUSE_S = 60
#: Words in an engine rejection that mean the conversation outgrew the context window: the model's doing.
CONTEXT_OVERFLOW_MARKERS = ("context length", "context_length", "maximum context", "longer than the maximum", "too long")

records_lock = threading.Lock()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
ports: queue_module.Queue[int] = queue_module.Queue()
ungradable: set[str] = set()


class InfrastructureFailure(RuntimeError):
    """A model call or the environment failed for a reason outside the model; retried, not scored."""


class VersionStraddle(RuntimeError):
    """The episode's turns came from more than one weight version; Reef cannot assemble it."""


class SampleTooLong(RuntimeError):
    """The assembled sample exceeds MAX_SAMPLE_TOKENS; the actor step cannot hold its logits."""


class Ungradable(RuntimeError):
    """AppWorld cannot grade this task (every requirement test passes for a do-nothing agent)."""


def load_problems() -> list[str]:
    path = Path(os.environ.get("SAO_PROBLEMS", str(APPWORLD_ROOT / "data" / "datasets" / "train.txt")))
    tasks = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not tasks:
        raise SystemExit(f"{path} lists no tasks")
    return tasks


def problem_order(problems: list[str], budget: int) -> list[str]:
    """Uniform without replacement within an epoch, reshuffled each epoch (the FinQA driver's order)."""
    rng = random.Random(SEED)
    order: list[str] = []
    while len(order) < budget:
        epoch = list(problems)
        rng.shuffle(epoch)
        order.extend(epoch)
    return order[:budget]


def releases() -> list[dict] | None:
    request = urllib.request.Request(
        f"{SERVICE_URL}/reef/scenarios/{SCENARIO}/releases", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())["releases"]
    except (urllib.error.URLError, TimeoutError):
        return None


def training_release_count() -> int | None:
    rows = releases()
    return None if rows is None else sum(1 for row in rows if row.get("operation") == "training")


def serving_release() -> str | None:
    rows = releases() or []
    current = [row for row in rows if row.get("current")]
    return str(current[0]["release_id"]) if current else None


def response_runtime_load_id(body: dict) -> str | None:
    for key in ("meta_info", "metadata"):
        meta = body.get(key)
        if isinstance(meta, dict) and meta.get("runtime_load_id") is not None:
            return str(meta["runtime_load_id"])
    for choice in body.get("choices") or []:
        meta = choice.get("meta_info") if isinstance(choice, dict) else None
        if isinstance(meta, dict) and meta.get("runtime_load_id") is not None:
            return str(meta["runtime_load_id"])
    return None


class ReefChat(ChatModel):
    """One AppWorld call through Reef: T=1.0 / top_p 1.0, the execute_code tool, one receipt per call."""

    def __init__(self, client: ReefClient, model: str) -> None:
        self.client = client
        self.model = model

    def complete(self, messages: list[dict], tools: list[dict], max_tokens: int) -> tuple[dict, dict]:
        # transformers 5 returns a BatchEncoding here, whose len() is its number of keys: count input_ids.
        encoded = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True,
                                                return_dict=True)
        prompt_tokens = len(encoded["input_ids"])
        room = CONTEXT_TOKENS - prompt_tokens - CONTEXT_MARGIN
        if room <= 0:
            raise ContextOverflow(f"prompt of {prompt_tokens} tokens leaves no room in a {CONTEXT_TOKENS}-token context")
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": min(max_tokens, room),
        }
        try:
            body, receipt = self.client.inference_with_record(SCENARIO, "/v1/chat/completions", payload)
        except ReefClientError as error:
            if any(marker in error.body.lower() for marker in CONTEXT_OVERFLOW_MARKERS):
                raise ContextOverflow(str(error)) from error
            raise InfrastructureFailure(str(error)) from error
        except (OSError, TimeoutError) as error:
            raise InfrastructureFailure(f"{type(error).__name__}: {error}") from error
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        return choice["message"], {
            "receipt": receipt,
            "runtime_load_id": response_runtime_load_id(body),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "prompt_tokens_local": prompt_tokens,
            "completion_tokens": usage.get("completion_tokens"),
            "max_tokens": payload["max_tokens"],
        }


def fetch_turn_record(agent_record_id: str) -> AgentRecord:
    request = urllib.request.Request(
        f"{SERVICE_URL}/reef/scenarios/{SCENARIO}/records/{agent_record_id}",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        raise InfrastructureFailure(f"record {agent_record_id} unreadable: {error}") from error
    return AgentRecord(
        agent_record_id=body["agent_record_id"],
        scenario=SCENARIO,
        request_type=RequestType(body["request_type"]),
        payload=body["payload"],
        created_at=float(body["created_at"]),
        references=tuple(body.get("references") or ()),
    )


def turn_versions(record: AgentRecord) -> set[str]:
    training = (record.payload.get("response") or {}).get("training") or {}
    spans = training.get("runtime_load_spans") or record.payload.get("runtime_load_spans") or []
    if spans:
        return {str(span["runtime_load_id"]) for span in spans}
    recorded = record.payload.get("runtime_load_id") or training.get("runtime_load_id")
    return {str(recorded)} if recorded else set()


def check_reef_source() -> None:
    """The assembly imported here must be the image's: the image tag names the Reef source commit."""
    image = os.environ.get("IMAGE", "")
    git = ["git", "-C", str(REEF_ROOT), "-c", "safe.directory=*"]
    commit = subprocess.run([*git, "log", "-1", "--format=%h", "--", ".", ":(exclude)exp_scripts"],
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run([*git, "status", "--porcelain", "--", ".", ":(exclude)exp_scripts"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if image != f"reef:sao-{commit}" or dirty:
        raise SystemExit(
            f"expected IMAGE=reef:sao-{commit} and a clean Reef source tree; got IMAGE={image!r}, "
            f"uncommitted changes: {dirty or 'none'}. The driver checks episodes with Reef's assembly code "
            f"from {REEF_ROOT}, which must be the code the stack runs: rebuild the image or check out its commit."
        )


def play(client: ReefClient, model: str, task_id: str, position: int) -> Episode:
    """One episode on its own AppWorld server; the server and the world it wrote are removed afterwards."""
    port = ports.get()
    experiment = f"sao-appworld/{RUN_TAG}/{position:06d}"
    server = AppWorldServer(port, APPWORLD_ROOT, APPWORLD_VENV, RECORDS_PATH.parent / "appworld_servers" / f"{port}.log")
    try:
        try:
            server.start()
        except RuntimeError as error:
            raise InfrastructureFailure(f"appworld server: {error}") from error
        try:
            return run_episode(ReefChat(client, model), server, task_id, experiment)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise InfrastructureFailure(f"appworld server request: {error}") from error
    finally:
        server.stop()
        shutil.rmtree(APPWORLD_ROOT / "experiments" / "outputs" / experiment, ignore_errors=True)
        ports.put(port)


def one_episode(client: ReefClient, model: str, task_id: str, position: int) -> dict:
    started = time.time()
    release = serving_release()
    episode = play(client, model, task_id, position)
    turns = [t for t in episode.turns if "receipt" in t]
    versions = {t.get("runtime_load_id") for t in turns}
    # As the FinQA driver: the report decision is Reef's own assembly of the turn records it holds.
    turn_records = [fetch_turn_record(t["receipt"]) for t in turns]
    split_turns = sum(len(turn_versions(r)) != 1 for r in turn_records)
    assemblable = bool(turn_records) and make_multi_turn_policy_trajectory(
        turn_records, 0.0, source_agent_record_id="driver-check",
        realign_threshold=REALIGN_THRESHOLD, scaffold_tolerance=SCAFFOLD_TOLERANCE,
    ) is not None
    straddled = None in versions or len(versions) > 1 or not assemblable
    # Every turn's prompt extends the previous one, so the last turn's tokens are the whole sample.
    sample_tokens = len(((turn_records[-1].payload.get("response") or {}).get("training") or {}).get("tokens") or []) if turn_records else 0
    record = {
        "position": position,
        "task_id": task_id,
        "turns": len(turns),
        "tool_calls": len(episode.dispatches),
        "stop_reason": episode.stop_reason,
        "reward": episode.reward,
        "tgc": episode.tgc,
        "verifier_status": episode.verifier_status,
        "finish_reasons": [t.get("finish_reason") for t in turns],
        "completion_tokens": episode.completion_tokens,
        "prompt_tokens_last": turns[-1].get("prompt_tokens") if turns else None,
        "runtime_load_ids": sorted(v for v in versions if v is not None),
        "split_turns": split_turns,
        "sample_tokens": sample_tokens,
        "serving_release_id": release,
        "agent_record_ids": [t["receipt"] for t in turns],
        "initial_state_sha256": episode.initial_state_sha256,
        "seconds": round(time.time() - started, 1),
    }
    if not turns:
        raise InfrastructureFailure(f"episode {task_id} ended before its first turn ({episode.stop_reason})")
    if episode.reward is None:
        ungradable.add(task_id)
        record.update(dropped="ungradable", recorded_at=time.time())
        write_record(record, episode)
        raise Ungradable(f"{task_id}: {episode.verifier_status}")
    if straddled:
        kind = ("version_missing" if None in versions else "version_split_turn" if split_turns
                else "version_straddle" if len(versions) > 1 else "unassemblable")
        record.update(dropped=kind, recorded_at=time.time())
        write_record(record, episode)
        raise VersionStraddle(f"{task_id} spans versions {record['runtime_load_ids'] or 'unknown'}")
    if sample_tokens > MAX_SAMPLE_TOKENS:
        record.update(dropped="too_long", recorded_at=time.time())
        write_record(record, episode)
        raise SampleTooLong(f"{task_id}: assembled sample of {sample_tokens} tokens > {MAX_SAMPLE_TOKENS}")
    client.report(SCENARIO, {"score": float(episode.reward), "references": record["agent_record_ids"]}, recipe=RECIPE)
    record.update(recorded_at=time.time())
    write_record(record, episode)
    print(
        f"[{RECIPE} {position}] {task_id} reward={episode.reward:.3f} tgc={episode.tgc} turns={record['turns']} "
        f"calls={record['tool_calls']} stop={episode.stop_reason} tokens={episode.completion_tokens} release={release}",
        flush=True,
    )
    return record


def write_record(record: dict, episode: Episode) -> None:
    with records_lock:
        RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RECORDS_PATH, "a") as out:
            out.write(json.dumps(record) + "\n")
        with open(EPISODES_PATH, "a") as out:
            out.write(json.dumps({"position": record["position"], "task_id": record["task_id"],
                                  "messages": episode.messages, "turns": episode.turns,
                                  "dispatches": episode.dispatches, "tracker": episode.tracker}) + "\n")


class TrainerPacer:
    """Hold new submissions while more than AHEAD batches are out beyond the trainer (the FinQA driver's)."""

    def __init__(self, progress_file: str | None) -> None:
        self.progress_file = progress_file
        self.started_at = time.time()
        self.credit = 0

    def trained_steps(self) -> tuple[int, float] | None:
        if not self.progress_file:
            return None
        try:
            with open(self.progress_file) as handle:
                steps = int(handle.read().strip() or 0)
            return steps, time.time() - os.path.getmtime(self.progress_file)
        except (OSError, ValueError):
            return 0, time.time() - self.started_at

    def wait(self, completed: int) -> None:
        while True:
            progress = self.trained_steps()
            if progress is None:
                return
            steps, idle = progress
            if completed < (steps + 1 + AHEAD) * BATCH + self.credit:
                return
            if idle > STALL_S:
                self.credit += BATCH
                print(f"pace: trainer idle {idle:.0f}s at step {steps}; releasing one more batch", flush=True)
                continue
            time.sleep(10)


def wait_for_training(expected: int) -> None:
    deadline = time.time() + TRAIN_DRAIN_TIMEOUT_S
    trained = None
    while time.time() < deadline:
        trained = training_release_count()
        if trained is not None and trained >= expected:
            print(f"trained: {trained}/{expected} steps committed", flush=True)
            return
        time.sleep(15)
    print(f"WARNING: only {trained}/{expected} steps trained within {TRAIN_DRAIN_TIMEOUT_S}s", flush=True)


def main() -> None:
    check_reef_source()
    if not (APPWORLD_VENV / "bin" / "appworld").is_file():
        raise SystemExit(f"expected the AppWorld CLI at {APPWORLD_VENV}/bin/appworld (APPWORLD_VENV)")
    problems = load_problems()
    order = problem_order(problems, BUDGET)
    for port in range(APPWORLD_PORT_BASE, APPWORLD_PORT_BASE + IN_FLIGHT):
        ports.put(port)
    client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=CALL_TIMEOUT_S)
    model = os.environ.get("SAO_MODEL_NAME", "reef")
    pacer = TrainerPacer(PROGRESS_FILE)
    print(
        f"pool={len(problems)} tasks, budget={BUDGET}, batch={BATCH}, in_flight={IN_FLIGHT}, "
        f"temperature={TEMPERATURE}, top_p={TOP_P}, context={CONTEXT_TOKENS}, appworld={APPWORLD_ROOT}",
        flush=True,
    )

    done = failures = dropped = streak = 0
    problems_by_future: dict[Future, str] = {}

    def settle(futures) -> tuple[list[str], list[str]]:
        nonlocal done, failures, dropped, streak
        retry_front, retry_back = [], []
        for future in futures:
            task_id = problems_by_future.pop(future)
            try:
                future.result()
            except VersionStraddle:
                dropped += 1
                retry_back.append(task_id)
            except Ungradable as error:
                dropped += 1
                print(f"ungradable, not retried: {error}", flush=True)
            except SampleTooLong as error:
                dropped += 1
                print(f"too long, not retried: {error}", flush=True)
            except (InfrastructureFailure, ReefClientError, OSError, KeyError, IndexError, TypeError) as error:
                failures += 1
                streak += 1
                retry_front.append(task_id)
                print(f"episode failed: {type(error).__name__}: {error}", flush=True)
            else:
                done += 1
                streak = 0
        if streak >= MAX_FAILURE_STREAK:
            raise SystemExit(f"{streak} episodes failed in a row; the serving stack is down")
        if retry_front:
            time.sleep(FAILURE_PAUSE_S)
        return retry_front, retry_back

    with ThreadPoolExecutor(max_workers=IN_FLIGHT) as pool:
        pending: set[Future] = set()
        queue = list(order)
        position = 0
        while done < BUDGET:
            while len(pending) < IN_FLIGHT and queue:
                pacer.wait(done)
                task_id = queue.pop(0)
                if task_id in ungradable:
                    continue
                future = pool.submit(one_episode, client, model, task_id, position)
                problems_by_future[future] = task_id
                pending.add(future)
                position += 1
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            retry_front, retry_back = settle(finished)
            queue = retry_front + queue + retry_back
    print(f"finished: {done} episodes reported, {dropped} dropped, {failures} failures", flush=True)
    wait_for_training(done // BATCH)


if __name__ == "__main__":
    main()
