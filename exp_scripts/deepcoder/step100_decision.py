"""Decide whether the lr5x DeepCoder run continues past step 100 (rule agreed 2026-09-26).

Continue only if step 100 is significantly better than base at T=0 or at T=1.0:
the 95% CI of the paired per-task difference (each side's 4-sample mean, 687 tasks)
has a lower bound above 0 on at least one side. Otherwise stop.

Exit codes: 0 = STOP, 10 = CONTINUE, 2 = results missing or incomplete.

usage: step100_decision.py [ITEM]   (default lr5x_step_100)
"""

from __future__ import annotations

import math
import statistics
import sys

from compare_t0_t1 import TASKS, item_samples, table

STOP, CONTINUE, INCOMPLETE = 0, 10, 2


def main() -> None:
    item = sys.argv[1] if len(sys.argv) > 1 else "lr5x_step_100"
    base, checkpoint = item_samples("base_29"), item_samples(item)
    if base is None or checkpoint is None:
        print(f"expected T=0 x4 and T=1.0 x4 results for base_29 and {item}; one is missing")
        sys.exit(INCOMPLETE)
    rows, significant = [], []
    for side, label in ((0, "T=0 (4/task)"), (1, "T=1.0 (4/task)")):
        if len(base[side]) != TASKS or len(checkpoint[side]) != TASKS:
            print(f"{label}: expected {TASKS} tasks per side, got base {len(base[side])} / {item} {len(checkpoint[side])}")
            sys.exit(INCOMPLETE)
        diffs = [sum(checkpoint[side][k]) / len(checkpoint[side][k]) - sum(base[side][k]) / len(base[side][k]) for k in base[side]]
        mean = statistics.mean(diffs)
        se = statistics.stdev(diffs) / math.sqrt(len(diffs))
        low, high = mean - 1.96 * se, mean + 1.96 * se
        significant.append(low > 0)
        for name, samples in (("Base", base), (item, checkpoint)):
            correct = sum(sum(v) for v in samples[side].values())
            total = sum(len(v) for v in samples[side].values())
            rows.append([label, name, f"{100 * correct / total:.2f}% ({correct}/{total})", f"{correct / total:.3f}",
                         "-" if name == "Base" else f"{100 * mean:+.2f} pt [{100 * low:+.2f}, {100 * high:+.2f}]"])
    table(["Sampling", "Model", "Success Rate", "Avg Score", "vs Base (paired, 95% CI)"], rows)
    if any(significant):
        print(f"DECISION: CONTINUE ({item} significantly above base at "
              f"{' and '.join(s for s, ok in zip(('T=0', 'T=1.0'), significant) if ok)})")
        sys.exit(CONTINUE)
    print(f"DECISION: STOP ({item} not significantly above base at T=0 nor at T=1.0)")
    sys.exit(STOP)


if __name__ == "__main__":
    main()
