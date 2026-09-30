"""FinQA multi-table SAO checkpoints vs base (Qwen3-4B-Instruct-2507), LoRA-served, per split.

finqa_singletable/compare_finqa.py for the multi-table line: greedy avg@4 (t0k4) and T=0.7 avg@4
(t07k4) side by side. Each cell: success rate (rubric score >= 0.9, numerator/denominator over
4 x N rollouts) and average rubric score; the difference is the paired per-task difference of
the 4-attempt mean score against the base under the same protocol, with a normal 95% CI.

The run is length-filtered SAO: samples > 16,384 tokens are not trained, and version-straddle
drops (re-queued) remove most long episodes; see length_filter_report.py. Epoch = step x 64 / 991,
counting the 10 critic-only steps.

usage: compare_multitable.py [RUN_TAG]   (default finqa-multitable-formal)
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "finqa_multitable-eval"
PROTOCOLS = (("t0k4", "Greedy avg@4"), ("t07k4", "T=0.7 avg@4"))
SPLITS = ("multi_val", "multi_test")


def load(path: Path) -> list[dict] | None:
    return json.loads(path.read_text())["items"] if path.exists() else None


def per_task(items: list[dict]) -> dict[int, float]:
    by: dict[int, list[float]] = defaultdict(list)
    for item in items:
        by[item["idx"]].append(item["reward"])
    return {k: statistics.mean(v) for k, v in by.items()}


def cell(items: list[dict] | None, base: list[dict] | None) -> tuple[str, str]:
    if items is None:
        return "pending", "pending"
    n, correct = len(items), sum(i["is_correct"] for i in items)
    head = f"{100 * correct / n:.1f}% ({correct}/{n}), {statistics.mean(i['reward'] for i in items):.3f}"
    if base is None or items is base:
        return head, "-"
    a, b = per_task(items), per_task(base)
    d = [a[k] - b[k] for k in a if k in b]
    m, se = statistics.mean(d), statistics.stdev(d) / math.sqrt(len(d))
    return head, f"{m:+.3f} [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}]"


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "finqa-multitable-formal"
    steps = sorted({int(m.group(1)) for p in (ROOT / run).glob("step_*_lora_*") if (m := re.match(r"step_(\d+)_lora_", p.name))})
    header = ["Split", "Checkpoint", "Epoch"]
    for _, label in PROTOCOLS:
        header += [f"{label} (SR, avg score)", f"{label.split()[0]} diff vs base (95% CI)"]
    rows = [header]
    for split in SPLITS:
        bases = {s: load(ROOT / f"base_{s}" / f"{split}.json") for s, _ in PROTOCOLS}
        for step in [0] + steps:
            row = [split, "base" if step == 0 else f"step {step}", "-" if step == 0 else f"{step * 64 / 991:.2f}"]
            for s, _ in PROTOCOLS:
                items = bases[s] if step == 0 else load(ROOT / run / f"step_{step:03d}_lora_{s}" / f"{split}.json")
                row += list(cell(items, bases[s]))
            rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    print("FinQA multi-table, length-filtered SAO (LoRA-served), 64 tasks per step, 1 rollout per task, train pool 991 tasks; "
          "SR = rubric score >= 0.9")
    print("\n".join([line(rows[0]), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in rows[1:]]))


if __name__ == "__main__":
    main()
