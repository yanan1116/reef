"""DeepCoder scoring for the SAO stream driver.

This is the rLLM DeepCoder verifier, unchanged: compatible_grader's
ForkserverEvaluator around cookbooks/deepcoder's deepcoder_evaluator, which runs
RewardCodeFn -> lcb_check_correctness_v2 against the row's hidden tests. So the
reward SAO trains on is byte-for-byte the reward the PRPO and GRPO DeepCoder
runs trained on, and the prompt is the flow's own SYSTEM_PROMPT.

The one addition is a bounded re-grade on a verifier TIMEOUT. On .24 the CPU
grader stalls under concurrency, and a timeout there scores working code 0.
Under GRPO those rollouts were dropped from the group; SAO has no group to drop
from, so the driver re-grades instead (the GRPO run measured ~96% of timeouts
clearing on a re-grade). A rollout still timing out after the re-grades scores
0 like any other failure and is flagged in the records.

Memory (measured 2026-09-23 on .24). The first formal run held every train row
as Python objects (_rows(), 34.7 GB resident) and graded all 128 in-flight
completions at once; parsing hidden tests of up to 153 MiB per task drove the
driver to 221 GB and Ray's memory monitor killed the training workers at 95%.
So grade() now reads ONE row from an Arrow table of the parquet the rLLM
registry points at, and at most GRADE_CONCURRENCY completions are graded at a
time. Generation keeps its full concurrency; grading takes seconds against a
minute of generation, so the cap costs no throughput.
"""

from __future__ import annotations

import functools
import os
import threading
from pathlib import Path
from types import SimpleNamespace

from compatible_grader import ForkserverEvaluator, _timed_out
from deepcoder_flow import SYSTEM_PROMPT  # re-exported for the driver

__all__ = ["SYSTEM_PROMPT", "grade", "question"]

TIMEOUT_REGRADES = 2
GRADE_CONCURRENCY = int(os.environ.get("DEEPCODER_GRADE_CONCURRENCY", "16"))
_grade_slots = threading.BoundedSemaphore(GRADE_CONCURRENCY)
_evaluator: ForkserverEvaluator | None = None
_evaluator_lock = threading.Lock()


@functools.lru_cache(maxsize=1)
def _table():
    """The train split as an Arrow table: column buffers, not 24k Python dicts."""
    import pyarrow.parquet as pq

    home = os.environ.get("RLLM_HOME")
    if not home:
        raise SystemExit("RLLM_HOME must point at deepcoder-run/runtime (the rLLM dataset registry)")
    path = Path(home) / "datasets" / "deepcoder" / "train.parquet"
    table = pq.read_table(path, memory_map=True)
    if table.num_rows != 24287:
        raise SystemExit(f"{path} has {table.num_rows} rows, expected 24287")
    return table


def _row(problem_idx: int) -> dict:
    return _table().slice(problem_idx, 1).to_pylist()[0]


@functools.lru_cache(maxsize=1)
def _rows() -> list[dict]:
    from rllm.data import DatasetRegistry

    dataset = DatasetRegistry.load_dataset("deepcoder", "train")
    if dataset is None:
        raise SystemExit("DeepCoder train split is not registered; set RLLM_HOME to deepcoder-run/runtime")
    rows = list(dataset.data)
    if len(rows) != 24287:
        raise SystemExit(f"DeepCoder train split has {len(rows)} rows, expected 24287")
    return rows


def question(problem_idx: int) -> str:
    return str(_row(problem_idx)["question"])


def grade(problem_idx: int, completion: str) -> tuple[float, dict]:
    """Score one completion against problem ``problem_idx``'s hidden tests."""
    global _evaluator
    with _evaluator_lock:
        if _evaluator is None:
            _evaluator = ForkserverEvaluator(drop_timeouts=False)
    episode = SimpleNamespace(artifacts={"answer": completion})
    regrades = 0
    with _grade_slots:
        row = _row(problem_idx)
        while True:
            out = _evaluator.evaluate(row, episode)
            timed_out = bool(_timed_out(getattr(out, "metadata", None)))
            if out.is_correct or not timed_out or regrades >= TIMEOUT_REGRADES:
                break
            regrades += 1
        del row
    correct = bool(out.is_correct)
    return (1.0 if correct else 0.0), {"grader_timeout": timed_out and not correct, "timeout_regrades": regrades}
