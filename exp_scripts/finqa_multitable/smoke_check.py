"""Check a FinQA multi-table SAO smoke run (smoke_then_formal.sh) before the formal run may start.

finqa/smoke_check.py with the benchmark swapped. Each check prints expected / actual; any
failure exits 1 and the formal run is not launched:

  trained        all 4 optimizer steps committed: multi-turn samples assembled and ingested
  reported       the budget of episodes was reported
  tool_use       the SGLang qwen25 parser returned tool calls (a wrong parser yields none)
  score          the judge ran and the mean rubric score is near the model's known level
                 (base, greedy, nano judge: 0.46 on multi_val+multi_test)
  straddle       episodes dropped for spanning a weight publication stay under 40% (single-table: 25%;
                 multi-table episodes run ~14 turns / ~9 min, about twice as long, so more of them
                 straddle a publication; the first smoke dropped 25.8%)
  rejected       episodes with a refused call (context overflow, 4xx) are rare
  token_count    the driver's local prompt count equals the engine's on >= 98% of turns (the context
                 check uses it; a rare difference only lets the engine refuse a call near the limit,
                 which the flow handles as a failed call)
  infra          infrastructure failures (retried, never reported) are rare
  adapters       every step's adapter was kept; the last one is non-zero (actor trained)
  full_bundle    the final full actor+critic bundle was kept

usage: smoke_check.py RESULTS_DIR KEEP_DIR STEPS BUDGET
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from safetensors import safe_open

FORMAL_STEPS = 155


def main() -> None:
    out, keep = Path(sys.argv[1]), Path(sys.argv[2])
    steps, budget = int(sys.argv[3]), int(sys.argv[4])
    records = [json.loads(line) for line in open(out / "records.jsonl") if line.strip()]
    reported = [r for r in records if "dropped" not in r]
    dropped = [r for r in records if r.get("dropped")]
    driver = (out / "driver.log").read_text()
    results: list[tuple[str, bool, str, str]] = []

    def check(name: str, ok: bool, expected: str, actual: str) -> None:
        results.append((name, ok, expected, actual))

    trained = re.search(r"trained: (\d+)/(\d+) steps committed", driver)
    check("trained", bool(trained) and int(trained.group(1)) >= steps, f">= {steps} steps committed",
          trained.group(0) if trained else "no 'trained:' line (drain timed out or driver died)")
    check("reported", len(reported) >= budget, f">= {budget}", str(len(reported)))
    if not reported:
        report(results)
    tools = sum(r["tool_calls"] > 0 for r in reported)
    check("tool_use", tools / len(reported) >= 0.8, ">= 80% of episodes call a tool",
          f"{100 * tools / len(reported):.1f}% ({tools}/{len(reported)})")
    mean_score = statistics.mean(r["score"] for r in reported)
    check("score", 0.25 <= mean_score <= 0.75, "mean rubric score in [0.25, 0.75]",
          f"{mean_score:.3f} over {len(reported)}; >= 0.9: {sum(r['is_correct'] for r in reported)}/{len(reported)}")
    total = len(reported) + len(dropped)
    check("straddle", len(dropped) / total <= 0.40, "<= 40% of finished episodes dropped",
          f"{100 * len(dropped) / total:.1f}% ({len(dropped)}/{total})")
    rejected = sum(bool(r["llm_errors"]) for r in reported)
    check("rejected", rejected / len(reported) <= 0.10, "<= 10% of reported episodes with a refused call",
          f"{100 * rejected / len(reported):.1f}% ({rejected}/{len(reported)})")
    mismatched = sum(r["prompt_count_mismatches"] for r in records)
    turns = sum(r["turns"] for r in records)
    check("token_count", mismatched <= 0.02 * turns, "local prompt count == engine count on >= 98% of turns",
          f"{mismatched}/{turns} turns differ")
    infra = driver.count("episode failed:")
    check("infra", infra <= 0.05 * len(reported), f"<= {0.05 * len(reported):.0f} retried failures", str(infra))

    missing = [s for s in range(1, steps + 1) if not (keep / "adapters" / f"hf_rollout_{s - 1:05d}" / "hf").is_dir()]
    last = keep / "adapters" / f"hf_rollout_{steps - 1:05d}" / "hf" / "adapter_model.safetensors"
    lora_b_max = 0.0
    if last.is_file():
        with safe_open(str(last), framework="pt") as tensors:
            for key in tensors.keys():
                if "lora_B" in key:
                    lora_b_max = max(lora_b_max, tensors.get_tensor(key).abs().max().item())
    check("adapters", not missing and lora_b_max > 0, f"steps 1-{steps} kept, step {steps} lora_B != 0",
          f"missing steps {missing or 'none'}, step {steps} max|lora_B| = {lora_b_max:.3g}")
    bundle = keep / "full" / f"step_{steps:03d}"
    parts = [p for p in ("hf", "actor", "critic") if (bundle / p).is_dir()]
    check("full_bundle", len(parts) == 3, f"{bundle}/{{hf,actor,critic}}", f"present: {parts}")

    print(f"[info] dropped before reporting (Reef could not assemble them): {dict(Counter(r['dropped'] for r in dropped))}")
    stamps = []
    for line in (out / "sidecar.log").read_text().splitlines():
        found = re.match(r"\[sidecar (\S+ \S+) \S+\] trained steps: (\d+)", line)
        if found and int(found.group(2)) >= 1:
            stamps.append(time.mktime(time.strptime(found.group(1), "%Y-%m-%d %H:%M:%S")))
    if len(stamps) >= 2:
        per_step = statistics.median(b - a for a, b in zip(stamps, stamps[1:])) / 60
        print(f"[info] median {per_step:.1f} min/step over steps 1-{len(stamps)}; "
              f"{FORMAL_STEPS} steps ~ {FORMAL_STEPS * per_step / 60 / 24:.1f} days")
    print(f"[info] turns mean {statistics.mean(r['turns'] for r in reported):.1f} / max {max(r['turns'] for r in reported)}, "
          f"completion tokens mean {statistics.mean(r['completion_tokens'] for r in reported):.0f}, "
          f"last prompt max {max(r['prompt_tokens_last'] or 0 for r in reported)}, "
          f"ended {dict(Counter(r['ended'] for r in reported))}, "
          f"fallback answers {sum(r['final_fallback_used'] for r in reported)}, "
          f"malformed tool calls {sum(r['malformed_tool_calls'] for r in reported)}, "
          f"tool errors {sum(r['tool_errors'] for r in reported)}/{sum(r['tool_calls'] for r in reported)}")
    report(results)


def report(results: list[tuple[str, bool, str, str]]) -> None:
    for name, ok, expected, actual in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<12} expected {expected}; actual {actual}")
    passed = all(ok for _, ok, _, _ in results)
    print("SMOKE PASSED" if passed else "SMOKE FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
