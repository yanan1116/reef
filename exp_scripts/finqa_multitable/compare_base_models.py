"""FinQA multi-table: Qwen3-4B-Instruct-2507 vs Qwen3.5-4B base models, same protocol.

Reads results/finqa_multitable-eval/{base,base_qwen35}_{t0k4,t07k4}/{multi_val,multi_test}.json
(eval_multitable.py, 4 attempts per task). Per split and sampling setting:

  main table   success rate (rubric score >= 0.9, numerator/denominator over 4 x N rollouts) and
               average rubric score for each model, plus the paired per-task difference of the
               4-attempt mean score (Qwen3.5 - 2507) with a normal 95% CI over tasks
  behaviour    mean turns and tool calls; endings (model_final / tool_turn_budget / context_reserve /
               tool_phase_llm_error); episodes with a refused call; fallback answers; mean table-access score

usage: compare_base_models.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "finqa_multitable-eval"
MODELS = (("Qwen3-4B-Instruct-2507", "base"), ("Qwen3.5-4B (no-think)", "base_qwen35"))
SAMPLINGS = (("t0k4", "Greedy avg@4"), ("t07k4", "T=0.7 avg@4"))
SPLITS = ("multi_val", "multi_test")


def load(prefix: str, sampling: str, split: str) -> dict | None:
    path = ROOT / f"{prefix}_{sampling}" / f"{split}.json"
    return json.loads(path.read_text()) if path.exists() else None


def per_task(items: list[dict]) -> dict[int, float]:
    by: dict[int, list[float]] = defaultdict(list)
    for item in items:
        by[item["idx"]].append(item["reward"])
    return {k: statistics.mean(v) for k, v in by.items()}


def table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    line = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(rows[0]), sep] + [line(r) for r in rows[1:]])


def main() -> None:
    main_rows = [["Split", "Sampling", "Model", "Success Rate", "Avg Score", "Diff vs 2507 (95% CI)"]]
    behaviour_rows = [["Split", "Sampling", "Model", "Turns", "Tool calls", "Endings (final/budget/ctx/llm_err)",
                       "Refused-call eps", "Fallback", "Table access"]]
    for split in SPLITS:
        for sampling, label in SAMPLINGS:
            reference = load("base", sampling, split)
            for name, prefix in MODELS:
                result = load(prefix, sampling, split)
                if result is None:
                    main_rows.append([split, label, name, "pending", "pending", "pending"])
                    continue
                items = result["items"]
                n, correct = len(items), sum(i["is_correct"] for i in items)
                avg = statistics.mean(i["reward"] for i in items)
                diff = "-"
                if prefix != "base" and reference is not None:
                    a, b = per_task(items), per_task(reference["items"])
                    d = [a[k] - b[k] for k in a if k in b]
                    m, se = statistics.mean(d), statistics.stdev(d) / math.sqrt(len(d))
                    diff = f"{m:+.3f} [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}]"
                main_rows.append([split, label, name, f"{100 * correct / n:.1f}% ({correct}/{n})", f"{avg:.3f}", diff])
                ends = Counter(i["ended"] for i in items)
                behaviour_rows.append([
                    split, label, name, f"{statistics.mean(i['turns'] for i in items):.1f}",
                    f"{statistics.mean(i['tool_calls'] for i in items):.1f}",
                    "/".join(str(ends.get(k, 0)) for k in ("model_final", "tool_turn_budget", "context_reserve", "tool_phase_llm_error")),
                    f"{sum(bool(i['llm_errors']) for i in items)}/{n}", f"{sum(i['final_fallback_used'] for i in items)}/{n}",
                    f"{statistics.mean(i['table_access'] or 0 for i in items):.3f}",
                ])
    print("FinQA multi-table base models (v2 protocol, gpt-5.4-nano rubric judge, 4 attempts per task; success = score >= 0.9)")
    print(table(main_rows))
    print()
    print("Behaviour (means over all rollouts; endings count rollouts)")
    print(table(behaviour_rows))


if __name__ == "__main__":
    main()
