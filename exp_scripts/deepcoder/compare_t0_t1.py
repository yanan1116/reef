"""Per-checkpoint scores, reported separately: T=0 (mean of 4 samples per task) and T=1.0 (mean of 4).

T=0 samples: the <item>_t0k4 run (4 attempts) if present, else the item's first four
single greedy runs (<item>, <item>_r2, ...), reused from the earlier scheme.
T=1.0 samples: the <item>_t1k4 run. Items: base_29 and lr5x_step_NNN; bf16-merged
variants (<item>_merged_t0k4 / _t1k4) are listed as their own rows.
Paired test: per-task difference of each side's 4-sample mean against base, 687 tasks.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "deepcoder" / "eval-c32-lora-lr5x"
TASKS = 687


def per_task(result: dict) -> dict[int, list[bool]]:
    samples: dict[int, list[bool]] = defaultdict(list)
    for item in result["items"]:
        samples[item["idx"]].append(bool(item["is_correct"]))
    return samples


def load(tag: str) -> dict | None:
    path = ROOT / tag / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def item_samples(item: str) -> tuple[dict[int, list[bool]], dict[int, list[bool]]] | None:
    t1 = load(f"{item}_t1k4")
    t0_run = load(f"{item}_t0k4")
    if t0_run is not None:
        t0 = per_task(t0_run)
    else:
        greedy = [load(item)] + [load(f"{item}_r{k}") for k in range(2, 12)]
        greedy = [g for g in greedy if g is not None][:4]
        if len(greedy) < 4:
            return None
        t0 = defaultdict(list)
        for run in greedy:
            for idx, values in per_task(run).items():
                t0[idx].extend(values)
    if t1 is None:
        return None
    return t0, per_task(t1)


def table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print("| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(row)) + " |")


def main() -> None:
    """Report T=0 and T=1.0 separately: each is the mean of 4 samples per task, paired against base."""
    names = sorted({re.sub(r"(_t[01]k4|_r\d+)$", "", p.name) for p in ROOT.iterdir() if (p / "result.json").exists()})
    items = [n for n in names if n == "base_29" or re.fullmatch(r"lr5x_step_\d{3}(_merged)?", n)]
    items.sort(key=lambda n: (0, 0, 0) if n == "base_29" else (1, int(n[10:13]), n.endswith("_merged")))
    loaded = {}
    for item in items:
        samples = item_samples(item)
        if samples is not None and all(len(side) == TASKS for side in samples):
            loaded[item] = samples
    for side, title in ((0, "T=0, 4 samples per task"), (1, "T=1.0, 4 samples per task")):
        base = {k: sum(v) / len(v) for k, v in loaded["base_29"][side].items()} if "base_29" in loaded else None
        rows = []
        for item, samples in loaded.items():
            per = samples[side]
            correct = sum(sum(v) for v in per.values())
            total = sum(len(v) for v in per.values())
            label = "Base" if item == "base_29" else f"step {int(item[10:13])}" + (" (merged)" if item.endswith("_merged") else "")
            row = [label, f"{100 * correct / total:.2f}% ({correct}/{total})", f"{correct / total:.3f}"]
            if base is None or item == "base_29":
                row += ["-", "-"]
            else:
                diffs = [sum(per[k]) / len(per[k]) - base[k] for k in base]
                mean = statistics.mean(diffs)
                se = statistics.stdev(diffs) / math.sqrt(len(diffs))
                row += [f"{100 * mean:+.2f} pt [{100 * (mean - 1.96 * se):+.2f}, {100 * (mean + 1.96 * se):+.2f}]", f"{mean / se:+.2f}"]
            rows.append(row)
        print(title)
        table(["Checkpoint", "Success Rate", "Avg Score", "vs Base (paired, 95% CI)", "z"], rows)
        print()


if __name__ == "__main__":
    main()
