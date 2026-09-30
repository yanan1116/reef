"""Stream FinQA multi-table episodes through Reef for SAO, N in flight, one sample per episode.

finqa/stream_finqa.py (the single-table SAO driver) with the benchmark swapped; pacing,
budget, receipt, assembly-check, drop and retry logic are unchanged. What differs:

* A rollout is a multi-table v2 episode: multitable_env.run_episode, which is rllm's
  multitable_v2_flow turn for turn (its system prompt and public-question builder, the 4
  tools, 8000-char tool-output cut, 45 tool-use turns then up to 5 final-synthesis attempts,
  budget-checkpoint and finalization messages, 90,000-char transcript reserve), with every
  model call sent through Reef's /v1/chat/completions for its receipt. The final-synthesis
  calls keep the tool list (see rllm_multitable/SOURCE.txt), so Reef can assemble the episode.
* Sampling matches rllm's multi-table training (train_16_multitable_grpo.sh): temperature 0.7,
  top_p 1.0; 2048 new tokens per tool-use call and 8192 per final call (the flow's own caps);
  a 49,152-token context. vLLM refused a call whose prompt plus max_tokens exceeded 49,152 with
  a 400 (20 such errors in 7,191 recorded rllm multi-table episodes); here the driver counts the
  prompt with the model's chat template and refuses the same call before sending it. A refused
  call is the flow's failed call: retried once, then the flow moves on (tool phase -> final
  synthesis; final phase -> next attempt), and the last non-empty reply is the fallback answer.
* The score is rllm's finqa_evaluator on a multi_table question: the gpt-5.4-nano rubric score
  in [0, 1] (is_correct = score >= 0.9), reported once per episode, referencing every turn's
  receipt in order, so sao_multiturn.MultiTurnSAORecipe assembles the turns into one sample.
* Failures: an infrastructure error (Reef/engine down, connection, timeout, a 5xx) is retried
  later without reporting, as in the FinQA driver.
* Synchronous batches (SAO_SYNC_BATCHES=1, the default here; decided 2026-09-30): the driver
  submits only the episodes one optimizer step needs (reported + in flight <= SAO_BATCH, a dropped
  or failed episode is topped up) and, once the batch is reported, waits for that training step to
  be published before sampling the next batch. Every batch is sampled by one weight version and
  nothing is in flight at a publication. The streaming mode (0: up to SAO_AHEAD batches ahead of
  the trainer) dropped 41% of multi-table episodes for spanning a publication, with a strong
  length bias (22% at 4-6k tokens, 90% above 12k); single-table streaming dropped 2.9%.
* Training budget (decided 2026-09-30, so that no episode is dropped): episodes run with
  multitable_env's token_budget = SAO_MAX_SAMPLE_TOKENS (16384, the single-table training
  seq-length). A tool-phase call that would leave less than one 2048-token reply plus a 4096-token
  final answer goes straight to final synthesis ("training_budget"), and final calls are capped,
  so every assembled sample fits and is trained. ~3-4% of episodes (the longest) finalize earlier
  than rllm's protocol; evaluation is unchanged. A sample that still exceeds the budget (should
  not happen) is dropped "too_long" and not re-queued, as before: The
  actor's fp32 [tokens, vocab] logits of one 31k-token sample needed 18.6 GiB and OOM'd the
  48 GB card (smoke 2026-09-29 16:39); ~0.8% of episodes are that long. The episode itself
  still runs under the full 49,152-token protocol.

Environment (defaults are the multi-table formal run):
  SAO_PROBLEMS       data/multi_train.jsonl from export_multitable.py (required)
  SAO_SCENARIO       Reef scenario (default sao-finqa-multitable)
  SAO_BATCH          episodes per optimizer step; = recipe batch-size (default 64)
  SAO_IN_FLIGHT      concurrent episodes (default 64)
  SAO_BUDGET         reported episodes before the driver stops (default 9920 = 155 steps x 64 = 10 epochs of 991)
  SAO_TEMPERATURE, SAO_TOP_P   per-call sampling (0.7, 1.0)
  SAO_CONTEXT_TOKENS served context (default 49152 = the recipe's context-length and rllm's max_model_len)
  SAO_MAX_SAMPLE_TOKENS  longest assembled sample reported for training (default 16384)
  SAO_SYNC_BATCHES   1 (default): sample each batch with one weight version; 0: stream as the FinQA driver
  SAO_MODEL_PATH     tokenizer used to count prompt tokens (default the 2507 HF snapshot)
  SAO_CALL_TIMEOUT_S one model call through Reef (default 1800)
  SAO_RECORDS_PATH   one JSON line per episode (reported or dropped)
  SAO_PROGRESS_FILE, SAO_AHEAD, SAO_STALL_S, SAO_SEED, SAO_TRAIN_DRAIN_TIMEOUT_S  as in deepcoder/stream.py
  FINQA_JUDGE_CREDS  judge credentials file (default exp_scripts/finqa/.judge_creds)
  FINQA_MULTI_TABLE_JUDGE_MODEL  default gpt-5.4-nano, as in rllm's multi-table training and evaluation
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
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
REEF_ROOT = HERE.parents[1]  # the reef fork checkout the image was built from
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "finqa_singletable"))
# Reef's assembly code, imported from the checkout; tomli_w (its one import this venv lacks) lives in
# exp_scripts/.reef-deps so the verbatim PRPO venv stays unchanged.
sys.path[1:1] = [str(REEF_ROOT), str(HERE.parent / ".reef-deps")]
from reef.core.records_types import AgentRecord, RequestType  # noqa: E402
from reef.train.processors.common import make_multi_turn_policy_trajectory  # noqa: E402
from judge_env import load_judge_env  # noqa: E402

os.environ.setdefault("FINQA_MULTI_TABLE_JUDGE_MODEL", "gpt-5.4-nano")  # before judge_env's default (mini)
load_judge_env()
from multitable_env import TOOL_SPECS, ChatModel, ModelCallRejected, MultiTableEpisode, grade, run_episode  # noqa: E402

SERVICE_URL = "http://127.0.0.1:8900"
TOKEN = "reef-local"
SCENARIO = os.environ.get("SAO_SCENARIO", "sao-finqa-multitable")
RECIPE = "sao"

BATCH = int(os.environ.get("SAO_BATCH", "64"))
IN_FLIGHT = int(os.environ.get("SAO_IN_FLIGHT", "64"))
BUDGET = int(os.environ.get("SAO_BUDGET", str(155 * 64)))
TEMPERATURE = float(os.environ.get("SAO_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("SAO_TOP_P", "1.0"))
CONTEXT_TOKENS = int(os.environ.get("SAO_CONTEXT_TOKENS", "49152"))
MAX_SAMPLE_TOKENS = int(os.environ.get("SAO_MAX_SAMPLE_TOKENS", "16384"))
SYNC_BATCHES = os.environ.get("SAO_SYNC_BATCHES", "1") == "1"
MODEL_PATH = os.environ.get(
    "SAO_MODEL_PATH",
    "/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
)
CALL_TIMEOUT_S = float(os.environ.get("SAO_CALL_TIMEOUT_S", "1800"))
RECORDS_PATH = Path(os.environ.get("SAO_RECORDS_PATH", "work/records/finqa_multitable.jsonl"))
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


class InfrastructureFailure(RuntimeError):
    """A model call failed for a reason outside the model; the episode is retried, not scored."""


class VersionStraddle(RuntimeError):
    """The episode's turns came from more than one weight version; Reef cannot assemble it."""


class SampleTooLong(RuntimeError):
    """The assembled sample exceeds MAX_SAMPLE_TOKENS; the actor step cannot hold its logits."""


def load_problems() -> list[dict]:
    path = os.environ.get("SAO_PROBLEMS")
    if not path:
        raise SystemExit("SAO_PROBLEMS must point at data/multi_train.jsonl (export_multitable.py)")
    with open(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 991 or any(row.get("split") != "multi_train" for row in rows):
        raise SystemExit(f"expected {path} to hold rllm's multi_train split (991 rows); got {len(rows)} rows")
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
    """One multi-table call through Reef: rllm's sampling, the four tools, one receipt per call."""

    def __init__(self, client: ReefClient, model: str) -> None:
        self.client = client
        self.model = model

    def prompt_tokens(self, messages: list[dict]) -> int:
        # transformers 5 returns a BatchEncoding here, whose len() is its number of keys: count input_ids.
        encoded = tokenizer.apply_chat_template(messages, tools=TOOL_SPECS, add_generation_prompt=True, tokenize=True,
                                                return_dict=True)
        return len(encoded["input_ids"])

    def complete(self, messages: list[dict], max_tokens: int) -> tuple[dict, dict]:
        prompt_tokens = self.prompt_tokens(messages)
        if prompt_tokens + max_tokens > CONTEXT_TOKENS:
            # vLLM's 400 in the rllm runs: "maximum context length is 49152 tokens. However, you requested ..."
            raise ModelCallRejected(f"prompt of {prompt_tokens} tokens + {max_tokens} output tokens exceeds the "
                                    f"{CONTEXT_TOKENS}-token context")
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOL_SPECS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": max_tokens,
        }
        try:
            body, receipt = self.client.inference_with_record(SCENARIO, "/v1/chat/completions", payload)
        except ReefClientError as error:
            model_caused = 400 <= error.status < 500 or any(m in error.body.lower() for m in CONTEXT_OVERFLOW_MARKERS)
            if model_caused:
                raise ModelCallRejected(str(error)) from error
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
            "max_tokens": max_tokens,
        }


def episode_versions(episode: MultiTableEpisode) -> set[str | None]:
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
    # InfrastructureFailure propagates: retried later. The training budget keeps every sample trainable.
    episode = run_episode(ReefChat(client, model), problem["task"], token_budget=MAX_SAMPLE_TOKENS)
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
    # Every turn's prompt extends the previous turn's prompt + reply, so the last turn's tokens
    # are the whole assembled sample.
    sample_tokens = len(((turn_records[-1].payload.get("response") or {}).get("training") or {}).get("tokens") or [])
    record = {
        "position": position,
        "problem_idx": problem["problem_idx"],
        "turns": len(episode.turns),
        "tool_calls": episode.tool_calls,
        "tool_errors": episode.tool_errors,
        "malformed_tool_calls": episode.malformed_tool_calls,
        "llm_errors": episode.llm_errors,
        "ended": episode.finalize_reason,
        "final_fallback_used": episode.final_fallback_used,
        "transcript_chars": episode.transcript_chars,
        # the context check counts prompts locally; the engine's count must agree
        "prompt_count_mismatches": sum(turn.info.get("prompt_tokens") != turn.info.get("prompt_tokens_local")
                                       for turn in episode.turns),
        "finish_reasons": [turn.info.get("finish_reason") for turn in episode.turns],
        "prompt_tokens_last": episode.turns[-1].info.get("prompt_tokens") if episode.turns else None,
        "completion_tokens": sum(turn.info.get("completion_tokens") or 0 for turn in episode.turns),
        "runtime_load_ids": sorted(v for v in versions if v is not None),
        "turns_without_version": sum(turn.info.get("runtime_load_id") is None for turn in episode.turns),
        "split_turns": split_turns,
        "sample_tokens": sample_tokens,
        "serving_release_id": release,
        "agent_record_ids": [turn.info["receipt"] for turn in episode.turns],
        "seconds": round(time.time() - started, 1),
    }
    if not episode.turns:
        raise InfrastructureFailure(f"episode ended before its first turn: {episode.llm_errors}")
    if straddled:
        kind = ("version_missing" if None in versions else "version_split_turn" if split_turns
                else "version_straddle" if len(versions) > 1 else "unassemblable")
        record.update(dropped=kind, recorded_at=time.time())
        write_record(record)
        raise VersionStraddle(f"problem {problem['problem_idx']} spans versions {record['runtime_load_ids'] or 'unknown'}")
    if sample_tokens > MAX_SAMPLE_TOKENS:
        record.update(dropped="too_long", recorded_at=time.time())
        write_record(record)
        raise SampleTooLong(f"problem {problem['problem_idx']}: assembled sample of {sample_tokens} tokens > {MAX_SAMPLE_TOKENS}")
    score, is_correct, grading = grade(problem["task"], episode)
    client.report(SCENARIO, {"score": score, "references": record["agent_record_ids"]}, recipe=RECIPE)
    record.update(score=score, is_correct=is_correct, table_access=grading.get("table_access"), recorded_at=time.time())
    write_record(record)
    print(
        f"[{RECIPE} {position}] idx={problem['problem_idx']} score={score:.3f} turns={record['turns']} "
        f"ended={episode.finalize_reason} tokens={record['completion_tokens']} release={release}",
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


def wait_for_publication(steps: int) -> None:
    """Block until ``steps`` training releases exist: the batch just reported has been trained and published."""
    if steps == 0:
        # The first batch samples the base; the scenario only exists once its first request arrives,
        # so asking for its releases before that returns 404 (sync smoke, 2026-09-30 11:14).
        return
    started = time.time()
    while True:
        trained = training_release_count()
        if trained is not None and trained >= steps:
            print(f"sync: step {steps} published after {time.time() - started:.0f}s; sampling batch {steps + 1}", flush=True)
            return
        if time.time() - started > TRAIN_DRAIN_TIMEOUT_S:
            raise SystemExit(f"sync: step {steps} not published within {TRAIN_DRAIN_TIMEOUT_S}s (trained={trained})")
        time.sleep(15)


def main() -> None:
    check_reef_source()
    problems = load_problems()
    order = problem_order(problems, BUDGET)
    client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=CALL_TIMEOUT_S)
    model = os.environ.get("SAO_MODEL_NAME", "reef")
    pacer = TrainerPacer(PROGRESS_FILE)
    print(
        f"pool={len(problems)} tasks, budget={BUDGET}, batch={BATCH}, in_flight={IN_FLIGHT}, "
        f"temperature={TEMPERATURE}, top_p={TOP_P}, context={CONTEXT_TOKENS}, "
        f"judge={os.environ['FINQA_MULTI_TABLE_JUDGE_MODEL']}, sync_batches={SYNC_BATCHES}",
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
            except SampleTooLong:
                dropped += 1  # not re-queued: the task would likely run long again
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
            room = IN_FLIGHT
            if SYNC_BATCHES:
                batch = done // BATCH
                if not pending and done % BATCH == 0:
                    wait_for_publication(batch)  # nothing in flight: the next batch starts on the new weights
                room = BATCH - (done - batch * BATCH) - len(pending)  # episodes this batch still needs
            while len(pending) < IN_FLIGHT and queue and room > 0:
                if not SYNC_BATCHES:
                    pacer.wait(done)
                room -= 1
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
    print(f"finished: {done} episodes reported, {dropped} dropped (version straddle / too long), {failures} failures", flush=True)
    wait_for_training(done // BATCH)


if __name__ == "__main__":
    main()
