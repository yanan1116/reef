"""FinQA SAO checkpoints vs base: greedy avg@4 and T=0.7 avg@4 side by side, per split.

greedy avg@4 (t0k4): the greedy protocol (T=0, seed 1234) run 4 times per task.
T=0.7 avg@4 (t07k4): T=0.7, 4 attempts per task. Both score the mean over the 4
attempts. Each difference is paired per task against the base under the same protocol:
the mean of per-task differences of 4-attempt means, with a normal 95% CI over tasks.
A cell whose run has not finished reads "pending".

Epoch = step x 64 / 4030 (the train pool), including the 10 critic-only warm-up steps.

usage: compare_finqa.py [RUN_TAG] [MODE]   (defaults finqa-b64-20260927T004448, lora)
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "finqa-eval"
TASKS_PER_STEP, TRAIN_TASKS = 64, 4030  # epoch = step * 64 / 4030 (one rollout per task, sampled without replacement)
PROTOCOLS = (("t0k4", "Greedy avg@4"), ("t07k4", "T=0.7 avg@4"))


def per_task(path: Path) -> tuple[dict[int, float], int, int] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    values: dict[int, list[float]] = defaultdict(list)
    for item in result["items"]:
        values[item["idx"]].append(float(item["is_correct"]))
    return {k: sum(v) / len(v) for k, v in values.items()}, result["correct"], result["total"]


def paired(checkpoint: dict[int, float], base: dict[int, float]) -> str:
    diffs = [checkpoint[k] - base[k] for k in base]
    mean = statistics.mean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    return f"{100 * mean:+.2f} [{100 * (mean - 1.96 * se):+.2f}, {100 * (mean + 1.96 * se):+.2f}]"


def table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print("| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(row)) + " |")


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "finqa-b64-20260927T004448"
    mode = sys.argv[2] if len(sys.argv) > 2 else "lora"
    steps = sorted({int(m.group(1)) for p in (ROOT / run).iterdir() if (m := re.fullmatch(rf"step_(\d{{3}})_{mode}_\w+", p.name))})
    rows = []
    for split in ("val", "test"):
        bases = {key: per_task(ROOT / f"base_{key}" / f"{split}.json") for key, _ in PROTOCOLS}
        row = [split, "base", "-"]
        for key, _ in PROTOCOLS:
            base = bases[key]
            row += ["pending", "-"] if base is None else [f"{100 * base[1] / base[2]:.1f}% ({base[1]}/{base[2]})", "-"]
        rows.append(row)
        for step in steps:
            row = [split, f"step {step}", f"{step * TASKS_PER_STEP / TRAIN_TASKS:.2f}"]
            for key, _ in PROTOCOLS:
                result = per_task(ROOT / run / f"step_{step:03d}_{mode}_{key}" / f"{split}.json")
                base = bases[key]
                if result is None:
                    row += ["pending", "pending"]
                else:
                    row += [f"{100 * result[1] / result[2]:.1f}% ({result[1]}/{result[2]})",
                            "pending" if base is None else paired(result[0], base[0])]
            rows.append(row)
    header = ["Split", "Checkpoint", "Epoch"]
    for _, name in PROTOCOLS:
        header += [name, f"{name.split()[0]} diff vs base (95% CI)"]
    print(f"FinQA SAO ({mode}-served), {TASKS_PER_STEP} distinct tasks per step, 1 rollout per task, "
          f"train pool {TRAIN_TASKS:,} tasks:")
    table(header, rows)


if __name__ == "__main__":
    main()
