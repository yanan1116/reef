"""Check a FinQA SAO smoke run (smoke_then_formal.sh) before the formal run may start.

Each check prints expected / actual; any failure exits 1 and the formal run is
not launched. What the smoke must show, and why each matters for 620 steps:

  trained        all 4 optimizer steps committed: multi-turn samples assembled and
                 ingested (an assembly error raises inside ingest and stalls training)
  reported       the budget of episodes was reported
  tool_use       the SGLang qwen25 parser returned tool calls (a wrong parser silently
                 yields 0 tool calls and every episode answers blind)
  score          the judge ran and the model is at its known level (base test 65.6%)
  straddle       episodes dropped for spanning a weight publication stay a minority
  call_failed    model-caused endings (4xx, context overflow, prompt > 8192) are rare
  infra          infrastructure failures (retried, never reported) are rare
  adapters       every step's adapter was kept; the last one is non-zero (actor trained)
  full_bundle    the final full actor+critic bundle was kept

Informational: whether replies carry runtime_load_id (else the straddle check falls
back to counting training releases), and the measured time per step.

usage: smoke_check.py RESULTS_DIR KEEP_DIR STEPS BUDGET
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    out, keep = Path(sys.argv[1]), Path(sys.argv[2])
    steps, budget = int(sys.argv[3]), int(sys.argv[4])
    records = [json.loads(line) for line in open(out / "records.jsonl") if line.strip()]
    reported = [r for r in records if "dropped" not in r]
    dropped = [r for r in records if r.get("dropped")]
    driver = (out / "driver.log").read_text()
    failed = [r for r in reported if r["ended"] == "call_failed"]
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
    check("tool_use", tools / len(reported) >= 0.5, ">= 50% of episodes call a tool",
          f"{100 * tools / len(reported):.1f}% ({tools}/{len(reported)})")
    correct = sum(r["score"] for r in reported)
    check("score", 0.3 <= correct / len(reported) <= 0.95, "train reward in [30%, 95%]",
          f"{100 * correct / len(reported):.1f}% ({correct:.0f}/{len(reported)})")
    total = len(reported) + len(dropped)
    check("straddle", len(dropped) / total <= 0.25, "<= 25% of finished episodes dropped",
          f"{100 * len(dropped) / total:.1f}% ({len(dropped)}/{total})")
    check("call_failed", len(failed) / len(reported) <= 0.10, "<= 10% of reported episodes",
          f"{100 * len(failed) / len(reported):.1f}% ({len(failed)}/{len(reported)})")
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

    kinds = {kind: sum(r.get("dropped") == kind for r in records)
             for kind in ("version_straddle", "version_split_turn", "version_missing", "unassemblable")}
    print(f"[info] dropped before reporting (Reef could not assemble them): {kinds}")
    visible = sum(bool(r["runtime_load_ids"]) for r in records)
    print(f"[info] runtime_load_id visible in {visible}/{len(records)} episode records"
          + ("" if visible else " -> straddle detection uses the training-release count"))
    stamps = []
    for line in (out / "sidecar.log").read_text().splitlines():
        found = re.match(r"\[sidecar (\S+ \S+) \S+\] trained steps: (\d+)", line)
        if found and int(found.group(2)) >= 1:
            stamps.append(time.mktime(time.strptime(found.group(1), "%Y-%m-%d %H:%M:%S")))
    if len(stamps) >= 2:
        per_step = statistics.median(b - a for a, b in zip(stamps, stamps[1:])) / 60
        print(f"[info] median {per_step:.1f} min/step over steps 1-{len(stamps)}; "
              f"620 steps ~ {620 * per_step / 60 / 24:.1f} days")
    turns = statistics.mean(r["turns"] for r in reported)
    print(f"[info] mean turns {turns:.2f}, mean completion tokens {statistics.mean(r['completion_tokens'] for r in reported):.0f}, "
          f"ended {dict((e, sum(r['ended'] == e for r in reported)) for e in ('answer', 'max_turns', 'call_failed', 'tool_error'))}")
    report(results)


def report(results: list[tuple[str, bool, str, str]]) -> None:
    for name, ok, expected, actual in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<12} expected {expected}; actual {actual}")
    passed = all(ok for _, ok, _, _ in results)
    print("SMOKE PASSED" if passed else "SMOKE FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
