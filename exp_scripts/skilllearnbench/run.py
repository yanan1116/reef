"""SAO training and evaluation on the SkillLearnBench split, through reef-eval's Harbor executor.

train  Every epoch runs each train instance SLB_REPEATS times, all shuffled together (seed
       SLB_SEED + epoch). One episode is one SAO sample (MultiTurnSAORecipe, as the AppWorld/FinQA SAO
       lines): the episode is reported once, referencing every model call's receipt in order, scored
       with its verifier ctrf test-pass fraction. Synchronous batches: SLB_BATCH episodes run on one
       weight version; when SLB_BATCH of them are reported the loop waits for that training step to
       publish before it starts the next batch, so no episode spans two versions. An episode that
       cannot be reported (no ctrf file, or longer than SLB_TRAIN_MAX_TOKENS) is replaced by the next
       one in the epoch's order. The training release count is recorded at every epoch end, so the
       checkpoint adapter of that epoch can be found.
eval   The test instances (SLB_EVAL_SPLIT=test|train|all), SLB_REPEATS times each, with the agent pointed at SLB_BASE_URL; nothing is
       reported. SLB_EVAL_TAG names the row group (e.g. base-T1, epoch3-T0).

    python run.py train | python run.py eval
Results: $SLB_RUN/episodes.jsonl (one line per trial), $SLB_RUN/epochs.jsonl (train).
"""

import asyncio
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

from reef_client import ReefClient
from reef_eval import Lab

# Reef's own multi-turn assembly, imported from the checkout (as the AppWorld/FinQA drivers), so an episode is
# reported only if the trainer can assemble it; tomli_w lives in exp_scripts/.reef-deps. Training only: eval runs
# also work from a copy of this directory without the Reef checkout (e.g. on .36).
if len(sys.argv) > 1 and sys.argv[1] == "train":
    REEF_ROOT = Path(__file__).resolve().parents[2]
    sys.path[1:1] = [str(REEF_ROOT), str(REEF_ROOT / "exp_scripts" / ".reef-deps")]
    from reef.core.records_types import AgentRecord, RequestType  # noqa: E402
    from reef.train.processors.common import make_multi_turn_policy_trajectory  # noqa: E402

REALIGN_THRESHOLD = 4096  # = recipe.config realign-threshold in configs/serve-skilllearnbench-*.yaml
SCAFFOLD_TOLERANCE = 0  # = sao_multiturn.MultiTurnSAORecipe scaffold_tolerance

HERE = Path(__file__).resolve().parent
TASKS = Path(os.environ["SLB_TASKS"])
RUN = Path(os.environ["SLB_RUN"])
REPEATS = int(os.environ.get("SLB_REPEATS", "8"))
EPOCHS = int(os.environ.get("SLB_EPOCHS", "10"))
SEED = int(os.environ.get("SLB_SEED", "20261014"))
CONCURRENCY = int(os.environ.get("SLB_CONCURRENCY", "16"))
TRAIN_MAX_TOKENS = int(os.environ.get("SLB_TRAIN_MAX_TOKENS", "32768"))
BATCH = int(os.environ.get("SLB_BATCH", "16"))  # episodes per optimizer step; = recipe batch-size
PUBLISH_TIMEOUT_S = float(os.environ.get("SLB_PUBLISH_TIMEOUT_S", "14400"))
ONLY = [name for name in os.environ.get("SLB_ONLY", "").split(",") if name]  # smoke: restrict instances
# SLB_AGENT=oracle (eval only; user, 2026-10-19 full-benchmark check) runs Harbor's oracle agent, which executes
# the task's solution/solve.sh: it checks that the environment builds and the verifier can pass, without a model.
AGENT_NAME = os.environ.get("SLB_AGENT", "harness:HarborAgent")
SERVICE_URL = os.environ.get("REEF_SERVICE_URL", "").rstrip("/")
SCENARIO = os.environ.get("REEF_SCENARIO", "slb-sao")
TOKEN = os.environ.get("REEF_TOKEN", "reef-local")


def trial_dir(uri: str | None) -> Path | None:
    return Path(unquote(urlparse(uri).path)) if uri else None


def ctrf_fraction(trial: Path) -> tuple[float | None, int, int]:
    """Passed / total tests from the verifier's ctrf.json, or ``None`` when there is none."""
    path = trial / "verifier" / "ctrf.json"
    try:
        summary = json.loads(path.read_text())["results"]["summary"]
    except (OSError, ValueError, KeyError):
        return None, 0, 0
    total = int(summary.get("tests") or 0)
    passed = int(summary.get("passed") or 0)
    return (passed / total if total else None), passed, total


def training_releases() -> int | None:
    request = urllib.request.Request(
        f"{SERVICE_URL}/reef/scenarios/{SCENARIO}/releases", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    return sum(1 for row in payload["releases"] if row.get("operation") == "training")


def turn_record(receipt: str) -> "AgentRecord":
    request = urllib.request.Request(
        f"{SERVICE_URL}/reef/scenarios/{SCENARIO}/records/{receipt}", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read())
    return AgentRecord(
        agent_record_id=body["agent_record_id"],
        scenario=SCENARIO,
        request_type=RequestType(body["request_type"]),
        payload=body["payload"],
        created_at=float(body["created_at"]),
        references=tuple(body.get("references") or ()),
    )


def turn_versions(record: "AgentRecord") -> set[str]:
    training = (record.payload.get("response") or {}).get("training") or {}
    spans = training.get("runtime_load_spans") or record.payload.get("runtime_load_spans") or []
    if spans:
        return {str(span["runtime_load_id"]) for span in spans}
    recorded = record.payload.get("runtime_load_id") or training.get("runtime_load_id")
    return {str(recorded)} if recorded else set()


def assembly_check(receipts: list[str]) -> tuple[str, int]:
    """("ok", sample tokens) when Reef can assemble the episode into one sample, else (reason, tokens)."""
    records = [turn_record(r) for r in receipts]
    versions = [turn_versions(r) for r in records]
    if any(len(v) != 1 for v in versions):
        return "version_missing_or_split", 0
    if len(set().union(*versions)) > 1:
        return "version_straddle", 0
    trajectory = make_multi_turn_policy_trajectory(
        records, 0.0, source_agent_record_id="driver-check",
        realign_threshold=REALIGN_THRESHOLD, scaffold_tolerance=SCAFFOLD_TOLERANCE,
    )
    tokens = len(((records[-1].payload.get("response") or {}).get("training") or {}).get("tokens") or [])
    return ("ok" if trajectory is not None else "unassemblable"), tokens


def report(client: ReefClient, trial: Path, instance: str, score: float, epoch: int) -> int:
    """One SAO report for the whole episode: its score against every turn's receipt, in call order.

    Returns the number of turns referenced, or 0 when the episode is not reportable (no turns, or
    its longest call exceeds SLB_TRAIN_MAX_TOKENS: the assembled sample would not fit the trainer).
    """
    result = json.loads((trial / "result.json").read_text())
    reef = ((result.get("agent_result") or {}).get("metadata") or {}).get("reef") or {}
    receipts = list(reef.get("agent_record_ids") or [])
    tokens = list(reef.get("agent_record_tokens") or [])
    if not receipts or (TRAIN_MAX_TOKENS and max(tokens or [0]) > TRAIN_MAX_TOKENS):
        return 0
    status, sample_tokens = assembly_check(receipts)
    if status != "ok":
        print(f"not reported: {trial.name} {status}", flush=True)
        log_line(RUN / "dropped.jsonl", {"trial": trial.name, "instance": instance, "epoch": epoch, "reason": status, "at": time.time()})
        return 0
    client.report(
        SCENARIO,
        {
            "agent_record_id": uuid.uuid5(uuid.NAMESPACE_URL, f"reef:slb:{trial.name}").hex,
            "score": score,
            "feedback": f"skilllearnbench {instance}: score {score:.3f} (ctrf test-pass fraction, else the 0/1 reward)",
            "references": receipts,
            "metadata": {"slb": {"instance": instance, "epoch": epoch, "trial": trial.name}},
        },
        recipe="sao",
    )
    return len(receipts)


def wait_for_step(target: int) -> int:
    """Block until the scenario has ``target`` training releases (the step of the last batch published)."""
    deadline = time.time() + PUBLISH_TIMEOUT_S
    while time.time() < deadline:
        count = training_releases()
        if count is not None and count >= target:
            return count
        time.sleep(15)
    raise SystemExit(f"training step {target} not published within {PUBLISH_TIMEOUT_S}s")


async def run_trials(lab: Lab, plan: list[dict], on_done, on_error=None) -> None:
    async def one(item: dict) -> None:
        agent = {"name": AGENT_NAME, "model_name": os.environ.get("SLB_MODEL", "reef")}
        try:
            row = await lab.run(str(TASKS / item["instance"]), agent, tags=item["tags"])
        except Exception as error:  # eval only: a task Harbor cannot load is recorded and the rest still run
            if on_error is None:
                raise
            await asyncio.to_thread(on_error, item, error)
            return
        await asyncio.to_thread(on_done, item, row)

    await asyncio.gather(*(one(item) for item in plan))


def log_line(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def summarize(item: dict, row) -> dict:
    trial = trial_dir(row.uri)
    fraction, passed, total = ctrf_fraction(trial) if trial else (None, 0, 0)
    slb = {}
    if trial and (trial / "result.json").is_file():
        result = json.loads((trial / "result.json").read_text())
        slb = ((result.get("agent_result") or {}).get("metadata") or {}).get("slb") or {}
    return {
        **item["tags"],
        "at": time.time(),
        "reward": (row.rewards or {}).get("reward"),
        "test_fraction": fraction,
        "tests_passed": passed,
        "tests_total": total,
        "error": row.tags.get("error") if row.tags else None,
        "trial": str(trial) if trial else None,
        **{f"agent_{k}": v for k, v in slb.items()},
    }


async def train() -> None:
    split = json.loads((TASKS / "split.json").read_text())
    instances = [name for name in split["train"] if not ONLY or name in ONLY]
    client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=600)
    lab = Lab(RUN / "lab", concurrency=CONCURRENCY)
    steps = training_releases() or 0
    reported_in_batch = 0  # carried across epochs: an epoch's tail fills the next epoch's first step
    for epoch in range(1, EPOCHS + 1):
        plan = [name for name in instances for _ in range(REPEATS)]
        random.Random(SEED + epoch).shuffle(plan)
        queue = [{"instance": n, "tags": {"mode": "train", "epoch": epoch, "slot": i, "instance": n}} for i, n in enumerate(plan)]
        log_line(RUN / "epochs.jsonl", {"epoch": epoch, "event": "start", "at": time.time(), "seed": SEED + epoch, "order": plan})

        def on_done(item: dict, row) -> None:
            nonlocal reported_in_batch
            record = summarize(item, row)
            trial = trial_dir(row.uri)
            record["batch_step"] = steps + 1
            # The score is the ctrf test-pass fraction; tasks whose verifier writes only reward.txt (temperature-simulation,
            # python-scala-translation) use that 0/1 reward instead (user, 2026-10-20: every training episode must be
            # usable for the update; the first full20 run silently skipped them). No score at all (the verifier failed)
            # stays unreported.
            score = record["test_fraction"] if record["test_fraction"] is not None else record["reward"]
            record["score"] = score
            record["reported_turns"] = report(client, trial, item["instance"], score, epoch) if trial and score is not None else 0
            reported_in_batch += record["reported_turns"] > 0
            log_line(RUN / "episodes.jsonl", record)
            print(f"epoch {epoch} step {steps + 1} slot {item['tags']['slot']} {item['instance']}: tests {record['tests_passed']}/{record['tests_total']} reward {record['reward']} turns {record['reported_turns']} end {record.get('agent_end', '')[:20]}", flush=True)

        while queue:
            # A batch: BATCH episodes on one weight version, topped up until BATCH are reported.
            while queue and reported_in_batch < BATCH:
                take, queue = queue[: BATCH - reported_in_batch], queue[BATCH - reported_in_batch :]
                await run_trials(lab, take, on_done)
            if reported_in_batch < BATCH:
                break  # the epoch's tail: fewer than BATCH reportable episodes left; they carry into the next step
            steps = wait_for_step(steps + 1)
            print(f"epoch {epoch}: training step {steps} published", flush=True)
            reported_in_batch = 0
        log_line(RUN / "epochs.jsonl", {"epoch": epoch, "event": "end", "at": time.time(), "training_releases": training_releases()})


async def evaluate() -> None:
    split = json.loads((TASKS / "split.json").read_text())
    # SLB_EVAL_SPLIT: test (default, the 9 held-out instances), train, or all (both).
    which = os.environ.get("SLB_EVAL_SPLIT", "test")
    pool = {"test": split["test"], "train": split["train"], "all": split["test"] + split["train"]}[which]
    instances = [name for name in pool if not ONLY or name in ONLY]
    tag = os.environ["SLB_EVAL_TAG"]
    lab = Lab(RUN / "lab", concurrency=CONCURRENCY)
    items = [
        {"instance": n, "tags": {"mode": "eval", "eval_tag": tag, "repeat": r, "instance": n}}
        for n in instances
        for r in range(REPEATS)
    ]

    def on_done(item: dict, row) -> None:
        record = summarize(item, row)
        log_line(RUN / "episodes.jsonl", record)
        print(f"{tag} {item['instance']} r{item['tags']['repeat']}: tests {record['tests_passed']}/{record['tests_total']} reward {record['reward']}", flush=True)

    def on_error(item: dict, error: Exception) -> None:
        log_line(RUN / "episodes.jsonl", {**item["tags"], "at": time.time(), "reward": None, "test_fraction": None,
                                          "tests_passed": 0, "tests_total": 0, "error": f"{type(error).__name__}: {error}"})
        print(f"{tag} {item['instance']} r{item['tags']['repeat']}: {type(error).__name__}", flush=True)

    await run_trials(lab, items, on_done, on_error)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "train":
        asyncio.run(train())
    elif mode == "eval":
        asyncio.run(evaluate())
    else:
        raise SystemExit("usage: run.py train|eval")
