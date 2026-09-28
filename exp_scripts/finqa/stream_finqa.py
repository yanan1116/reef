"""Stream FinQA episodes through Reef for SAO, N in flight, one sample per episode.

The streaming, pacing and budget logic is deepcoder/stream.py's (itself
recipes/sao/examples/imo_answerbench/stream.py plus task edits). What differs:

* A rollout is a multi-turn FinQA episode: finqa_env.run_episode, which is rllm's
  finqa_flow turn for turn (same system prompt, 4 tools, 8000-char tool-output
  cut, 20 turns), with every model call sent through Reef's
  /v1/chat/completions for its receipt.
* Sampling matches the PRPO FinQA runs: temperature 0.7, top_p 1.0, and at most
  2048 new tokens per turn (PRPO's data.max_response_length). A turn whose prompt
  exceeds 8192 tokens (PRPO's data.max_prompt_length) ends the episode with no
  answer, as rllm's gateway refused such a call; that turn is not trained. The recipe's
  rollout-temperature must be the same 0.7, since SAO's DIS ratio compares the
  training policy against these sampled log-probs.
* The score is rllm's finqa_evaluator (gpt-5.4-nano judge, correctness only).
  It is reported once per episode, referencing every turn's receipt in order, so
  sao_multiturn.MultiTurnSAORecipe assembles the turns into one sample.
* An episode Reef cannot assemble into one sample stalls training for good (the
  SAO processor raises inside ingest). So before reporting, the driver reads
  every turn's record back (GET /reef/scenarios/<s>/records/<id>) and runs
  Reef's own make_multi_turn_policy_trajectory on them. An episode it cannot
  assemble -- turns from different weight versions, a turn generated across a
  publication, a reply without a version, a token fork -- is not reported: it
  is recorded as dropped and its task goes back into the queue.
* Failures: an infrastructure error (Reef/engine down, connection, timeout) is
  retried later without reporting, as in the DeepCoder driver; a request the
  model itself caused to fail (a 4xx, or a context-length rejection) ends the
  episode with no answer and is scored 0 and reported, as finqa_flow scored it
  in the PRPO runs.

Environment (defaults are the FinQA formal run):
  SAO_PROBLEMS       finqa_train.jsonl from export_finqa.py (required)
  SAO_SCENARIO       Reef scenario (default sao-finqa)
  SAO_BATCH          episodes per optimizer step; = recipe batch-size (default 64)
  SAO_IN_FLIGHT      concurrent episodes (default 64)
  SAO_BUDGET         reported episodes before the driver stops (default 39680 = 620 steps x 64 = 10 epochs)
  SAO_TEMPERATURE, SAO_TOP_P, SAO_MAX_TOKENS   per-turn sampling (0.7, 1.0, 2048)
  SAO_MAX_PROMPT_TOKENS  a longer turn prompt ends the episode unanswered (8192)
  SAO_CALL_TIMEOUT_S one model call through Reef (default 1800)
  SAO_RECORDS_PATH   one JSON line per episode (reported or dropped)
  SAO_PROGRESS_FILE, SAO_AHEAD, SAO_STALL_S, SAO_SEED, SAO_TRAIN_DRAIN_TIMEOUT_S  as in deepcoder/stream.py
  FINQA_JUDGE_CREDS  judge credentials file (default exp_scripts/finqa/.judge_creds)
"""

from __future__ import annotations

import json
import os
import random
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
# Reef's assembly code, imported from the checkout; tomli_w (its one import this venv lacks) lives in
# exp_scripts/.reef-deps so the verbatim PRPO venv stays unchanged.
sys.path[1:1] = [str(REEF_ROOT), str(HERE.parent / ".reef-deps")]
from reef.core.records_types import AgentRecord, RequestType  # noqa: E402
from reef.train.processors.common import make_multi_turn_policy_trajectory  # noqa: E402
from judge_env import load_judge_env  # noqa: E402

load_judge_env()
from finqa_env import TOOL_SPECS, ChatModel, EpisodeResult, grade, run_episode  # noqa: E402

SERVICE_URL = "http://127.0.0.1:8900"
TOKEN = "reef-local"
SCENARIO = os.environ.get("SAO_SCENARIO", "sao-finqa")
RECIPE = "sao"

BATCH = int(os.environ.get("SAO_BATCH", "64"))
IN_FLIGHT = int(os.environ.get("SAO_IN_FLIGHT", "64"))
BUDGET = int(os.environ.get("SAO_BUDGET", str(620 * 64)))
TEMPERATURE = float(os.environ.get("SAO_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("SAO_TOP_P", "1.0"))
MAX_TOKENS = int(os.environ.get("SAO_MAX_TOKENS", "2048"))
MAX_PROMPT_TOKENS = int(os.environ.get("SAO_MAX_PROMPT_TOKENS", "8192"))
CALL_TIMEOUT_S = float(os.environ.get("SAO_CALL_TIMEOUT_S", "1800"))
RECORDS_PATH = Path(os.environ.get("SAO_RECORDS_PATH", "work/records/finqa.jsonl"))
SEED = int(os.environ.get("SAO_SEED", "0"))
TRAIN_DRAIN_TIMEOUT_S = int(os.environ.get("SAO_TRAIN_DRAIN_TIMEOUT_S", "14400"))
PROGRESS_FILE = os.environ.get("SAO_PROGRESS_FILE")
AHEAD = int(os.environ.get("SAO_AHEAD", "3"))
STALL_S = int(os.environ.get("SAO_STALL_S", "2700"))
REALIGN_THRESHOLD = 1024  # = sao_multiturn.MultiTurnSAORecipe realign_threshold
SCAFFOLD_TOLERANCE = 0    # = sao_multiturn.MultiTurnSAORecipe scaffold_tolerance
MAX_FAILURE_STREAK = 24
FAILURE_PAUSE_S = 60
#: Words in an engine rejection that mean the conversation outgrew the context window: the model's doing.
CONTEXT_OVERFLOW_MARKERS = ("context length", "context_length", "maximum context", "longer than the maximum", "too long")

records_lock = threading.Lock()


class InfrastructureFailure(RuntimeError):
    """A model call failed for a reason outside the model; the episode is retried, not scored."""


class PromptTooLong(RuntimeError):
    """A turn's prompt outgrew PRPO's max_prompt_length: the model's doing, scored like a refused call."""


class VersionStraddle(RuntimeError):
    """The episode's turns came from more than one weight version; Reef cannot assemble it."""


def load_problems() -> list[dict]:
    path = os.environ.get("SAO_PROBLEMS")
    if not path:
        raise SystemExit("SAO_PROBLEMS must point at finqa_train.jsonl (export_finqa.py)")
    with open(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(row.get("split") != "train" for row in rows):
        raise SystemExit(f"{path} must hold the train split only")
    return rows


def problem_order(problems: list[dict], budget: int) -> list[dict]:
    """Uniform without replacement within an epoch, reshuffled each epoch."""
    rng = random.Random(SEED)
    order: list[dict] = []
    while len(order) < budget:
        epoch = list(problems)
        rng.shuffle(epoch)
        order.extend(epoch)
    return order[:budget]


def releases() -> list[dict] | None:
    """The scenario's release chain, or None while the service cannot answer."""
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
    """The release marked current; deepcoder/stream.py's rows[-1] named the oldest one."""
    rows = releases() or []
    current = [row for row in rows if row.get("current")]
    return str(current[0]["release_id"]) if current else None


def response_runtime_load_id(body: dict) -> str | None:
    """The engine weight version that answered, where Reef leaves it in the reply (see reef/surface/weights.py)."""
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
    """One FinQA turn through Reef: the PRPO sampling, the four tools, one receipt per call."""

    def __init__(self, client: ReefClient, model: str) -> None:
        self.client = client
        self.model = model

    def complete(self, messages: list[dict]) -> tuple[dict, dict]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOL_SPECS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
        }
        try:
            body, receipt = self.client.inference_with_record(SCENARIO, "/v1/chat/completions", payload)
        except ReefClientError as error:
            model_caused = 400 <= error.status < 500 or any(m in error.body.lower() for m in CONTEXT_OVERFLOW_MARKERS)
            if model_caused:
                raise
            raise InfrastructureFailure(str(error)) from error
        except (OSError, TimeoutError) as error:
            raise InfrastructureFailure(f"{type(error).__name__}: {error}") from error
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        if (usage.get("prompt_tokens") or 0) > MAX_PROMPT_TOKENS:
            raise PromptTooLong(f"prompt of {usage['prompt_tokens']} tokens exceeds {MAX_PROMPT_TOKENS}")
        return choice["message"], {
            "receipt": receipt,
            "runtime_load_id": response_runtime_load_id(body),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }


def episode_versions(episode: EpisodeResult) -> set[str | None]:
    return {turn.info.get("runtime_load_id") for turn in episode.turns}


def fetch_turn_record(agent_record_id: str) -> AgentRecord:
    """One turn's record as Reef stored it for training (tokens, loss mask, per-token versions)."""
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
    """The weight versions of one turn's tokens: its runtime_load_spans, else its single recorded version."""
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


def one_episode(client: ReefClient, model: str, problem: dict, position: int) -> dict:
    started = time.time()
    release = serving_release()
    episode = run_episode(ReefChat(client, model), problem["task"]["question"])
    if isinstance(episode.failure, InfrastructureFailure):
        raise episode.failure  # retried later; nothing is reported
    versions = episode_versions(episode)
    # Reef assembles only turns that all name one weight version (make_multi_turn_policy_trajectory
    # returns None for a turn without runtime_load_id, and the SAO processor then raises inside ingest,
    # stalling training for good). Turns served around a publication can come back without one
    # (smoke 2026-09-26: 3 of 256 episodes, positions 238-240, at the step-2 publish), so such an
    # episode is dropped like a straddle; counting training releases instead let those three through.
    # A turn can also hold tokens of two versions while its reply names only the last one
    # (smoke 2026-09-26 22:27: position 235, turn 0 tokens 0-70 on :0, 70-148 on :1), and any
    # episode Reef cannot assemble stalls training for good. So the decision is Reef's own:
    # read every turn's record back and run the assembly the SAO processor will run.
    turn_records = [fetch_turn_record(turn.info["receipt"]) for turn in episode.turns]
    split_turns = sum(len(turn_versions(r)) != 1 for r in turn_records)
    assemblable = bool(turn_records) and make_multi_turn_policy_trajectory(
        turn_records, 0.0, source_agent_record_id="driver-check",
        realign_threshold=REALIGN_THRESHOLD, scaffold_tolerance=SCAFFOLD_TOLERANCE,
    ) is not None
    straddled = None in versions or len(versions) > 1 or not assemblable
    record = {
        "position": position,
        "problem_idx": problem["problem_idx"],
        "turns": len(episode.turns),
        "tool_calls": sum(turn.tool_calls for turn in episode.turns),
        "ended": episode.ended,
        "failure": f"{type(episode.failure).__name__}: {episode.failure}" if episode.failure else None,
        "finish_reasons": [turn.info.get("finish_reason") for turn in episode.turns],
        "prompt_tokens_last": episode.turns[-1].info.get("prompt_tokens") if episode.turns else None,
        "completion_tokens": sum(turn.info.get("completion_tokens") or 0 for turn in episode.turns),
        "runtime_load_ids": sorted(v for v in versions if v is not None),
        "turns_without_version": sum(turn.info.get("runtime_load_id") is None for turn in episode.turns),
        "split_turns": split_turns,
        "serving_release_id": release,
        "agent_record_ids": [turn.info["receipt"] for turn in episode.turns],
        "seconds": round(time.time() - started, 1),
    }
    if not episode.turns:
        raise InfrastructureFailure(f"episode ended before its first turn: {record['failure']}")
    if straddled:
        kind = ("version_missing" if None in versions else "version_split_turn" if split_turns
                else "version_straddle" if len(versions) > 1 else "unassemblable")
        record.update(dropped=kind, recorded_at=time.time())
        write_record(record)
        raise VersionStraddle(f"problem {problem['problem_idx']} spans versions {record['runtime_load_ids'] or 'unknown'}")
    score, is_correct, grading = grade(problem["task"], episode)
    client.report(SCENARIO, {"score": score, "references": record["agent_record_ids"]}, recipe=RECIPE)
    record.update(score=score, is_correct=is_correct, table_access=grading.get("table_access"), recorded_at=time.time())
    write_record(record)
    print(
        f"[{RECIPE} {position}] idx={problem['problem_idx']} score={score:.0f} turns={record['turns']} "
        f"ended={episode.ended} tokens={record['completion_tokens']} release={release}",
        flush=True,
    )
    return record


def write_record(record: dict) -> None:
    with records_lock:
        RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RECORDS_PATH, "a") as out:
            out.write(json.dumps(record) + "\n")


class TrainerPacer:
    """Hold new submissions while more than AHEAD batches are out beyond the trainer (deepcoder/stream.py)."""

    def __init__(self, progress_file: str | None) -> None:
        self.progress_file = progress_file
        self.started_at = time.time()
        self.credit = 0

    def trained_steps(self) -> tuple[int, float] | None:
        """(steps done, seconds since the progress file last changed); None when pacing is off."""
        if not self.progress_file:
            return None
        try:
            with open(self.progress_file) as handle:
                steps = int(handle.read().strip() or 0)
            return steps, time.time() - os.path.getmtime(self.progress_file)
        except (OSError, ValueError):
            return 0, time.time() - self.started_at

    def wait(self, completed: int) -> None:
        """Block until one more episode fits within AHEAD batches of the trainer's progress."""
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
    """Block until ``expected`` training releases exist, so the last step is not cut off."""
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
    problems = load_problems()
    order = problem_order(problems, BUDGET)
    client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=CALL_TIMEOUT_S)
    model = os.environ.get("SAO_MODEL_NAME", "reef")
    pacer = TrainerPacer(PROGRESS_FILE)
    print(
        f"pool={len(problems)} tasks, budget={BUDGET}, batch={BATCH}, in_flight={IN_FLIGHT}, "
        f"temperature={TEMPERATURE}, top_p={TOP_P}, max_tokens/turn={MAX_TOKENS}",
        flush=True,
    )

    done = failures = dropped = streak = 0
    problems_by_future: dict[Future, dict] = {}

    def settle(futures) -> tuple[list[dict], list[dict]]:
        """Count finished episodes: (retry now, retry at the back). Failures are never spent from the budget."""
        nonlocal done, failures, dropped, streak
        retry_front, retry_back = [], []
        for future in futures:
            problem = problems_by_future.pop(future)
            try:
                future.result()
            except VersionStraddle:
                dropped += 1
                retry_back.append(problem)  # keep the epoch's coverage; revisit it later
            except (InfrastructureFailure, ReefClientError, OSError, KeyError, IndexError, TypeError) as error:
                failures += 1
                streak += 1
                retry_front.append(problem)
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
                problem = queue.pop(0)
                future = pool.submit(one_episode, client, model, problem, position)
                problems_by_future[future] = problem
                pending.add(future)
                position += 1
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            retry_front, retry_back = settle(finished)
            queue = retry_front + queue + retry_back
    print(f"finished: {done} episodes reported, {dropped} dropped (version straddle), {failures} failures", flush=True)
    wait_for_training(done // BATCH)


if __name__ == "__main__":
    main()
