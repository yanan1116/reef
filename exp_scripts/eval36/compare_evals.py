"""Paired comparison of two evaluations of the same tasks (FinQA single/multi-table result.json files).

usage: compare_evals.py A.json B.json [--metric is_correct|reward] [--label-a X --label-b Y]

Per task, each side is the mean over its attempts; the difference B - A is tested over tasks with
  - a sign-flip permutation test (two-sided, 20000 draws; exact for the mean difference, no normality),
  - a bootstrap 95% CI over tasks (20000 resamples),
  - for single-attempt binary sides, McNemar's exact test on the discordant tasks.
Numbers only use numpy (the eval venv has no scipy).
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np


def per_task(path: str, metric: str) -> tuple[dict, int]:
    data = json.load(open(path))
    values = defaultdict(list)
    for item in data["items"]:
        value = item[metric]
        values[str(item["idx"])].append(float(value) if not isinstance(value, bool) else float(value))
    attempts = max(len(v) for v in values.values())
    return {task: sum(v) / len(v) for task, v in values.items()}, attempts


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--metric", default="is_correct")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()
    a, ka = per_task(args.a, args.metric)
    b, kb = per_task(args.b, args.metric)
    if set(a) != set(b):
        raise SystemExit(f"task sets differ: {len(set(a) ^ set(b))} tasks in only one side")
    tasks = sorted(a)
    x, y = np.array([a[t] for t in tasks]), np.array([b[t] for t in tasks])
    d = y - x
    rng = np.random.default_rng(0)
    observed = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(20000, len(d)))
    p_perm = float((np.abs((flips * d).mean(axis=1)) >= abs(observed) - 1e-12).mean())
    boot = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    line = (f"{args.label_a} (k={ka}) {x.mean():.4f}  vs  {args.label_b} (k={kb}) {y.mean():.4f}  | n={len(tasks)} tasks"
            f"  diff {observed:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  sign-flip p={p_perm:.4f}"
            f"  tasks changed {int((d != 0).sum())}")
    if ka == kb == 1 and args.metric == "is_correct":
        only_b, only_a = int(((y == 1) & (x == 0)).sum()), int(((x == 1) & (y == 0)).sum())
        line += f"  McNemar: {args.label_a}-only {only_a}, {args.label_b}-only {only_b}, p={mcnemar_exact(only_a, only_b):.4f}"
    print(line)


if __name__ == "__main__":
    main()
