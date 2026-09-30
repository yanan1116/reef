"""Check an AppWorld SAO smoke run (smoke_then_formal.sh) before the formal run may start.

The FinQA smoke checks with the benchmark swapped. Each check prints expected / actual; any
failure exits 1 and the formal run is not launched.

  trained        all STEPS optimizer steps committed (assembled multi-turn samples ingested)
  reported       the budget of episodes was reported
  graded         the grader ran: some episode scored above 0 (base 2507 train90: 13.3% do)
  token_count    the driver's local prompt-token count matches the engine's (the context
                 clamp on max_tokens depends on it): max |local - engine| <= 64
  overflow       context_overflow endings stay rare (<= 5% of reported episodes)
  straddle       episodes dropped before reporting stay a minority (<= 25%)
  infra          retried infrastructure failures are rare (<= 5% of reported)
  cleanup        no AppWorld server left on the driver's ports; no episode world left on disk
  adapters       every step's adapter kept; the last one non-zero (the actor trained)
  full_bundle    the final full actor+critic bundle kept

usage: smoke_check.py RESULTS_DIR KEEP_DIR STEPS BUDGET APPWORLD_ROOT RUN_TAG PORT_BASE IN_FLIGHT
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
    appworld_root, run_tag = Path(sys.argv[5]), sys.argv[6]
    port_base, in_flight = int(sys.argv[7]), int(sys.argv[8])
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
    positive = sum((r["reward"] or 0) > 0 for r in reported)
    check("graded", positive > 0, "some episode scores above 0 (base: 13.3% of train90)",
          f"{100 * positive / len(reported):.1f}% ({positive}/{len(reported)}) above 0, "
          f"mean {statistics.mean(r['reward'] for r in reported):.3f}")
    gaps = []
    for line in open(out / "episodes.jsonl"):
        for turn in json.loads(line)["turns"]:
            if turn.get("prompt_tokens") is not None and turn.get("prompt_tokens_local") is not None:
                gaps.append(abs(turn["prompt_tokens"] - turn["prompt_tokens_local"]))
    check("token_count", bool(gaps) and max(gaps) <= 64, "max |local - engine| prompt tokens <= 64",
          f"max {max(gaps) if gaps else 'n/a'} over {len(gaps)} turns")
    overflow = sum(r["stop_reason"] == "context_overflow" for r in reported)
    check("overflow", overflow <= 0.05 * len(reported), "<= 5% context_overflow",
          f"{100 * overflow / len(reported):.1f}% ({overflow}/{len(reported)})")
    total = len(reported) + len(dropped)
    check("straddle", len(dropped) / total <= 0.25, "<= 25% dropped before reporting",
          f"{100 * len(dropped) / total:.1f}% ({len(dropped)}/{total})")
    infra = driver.count("episode failed:")
    check("infra", infra <= 0.05 * len(reported), f"<= {0.05 * len(reported):.0f} retried failures", str(infra))
    left_worlds = list((appworld_root / "experiments" / "outputs" / "sao-appworld" / run_tag).glob("*"))
    left_servers = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        found = re.search(r"appworld serve environment --port (\d+)", cmd)
        if found and port_base <= int(found.group(1)) < port_base + in_flight:
            left_servers.append(proc.name)
    check("cleanup", not left_worlds and not left_servers, "no servers on the driver's ports, no worlds on disk",
          f"{len(left_servers)} servers, {len(left_worlds)} worlds left")

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

    kinds: dict[str, int] = {}
    for r in dropped:
        kinds[r["dropped"]] = kinds.get(r["dropped"], 0) + 1
    print(f"[info] dropped before reporting: {kinds}")
    stops: dict[str, int] = {}
    for r in reported:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
    print(f"[info] endings: {stops}; mean turns {statistics.mean(r['turns'] for r in reported):.1f}, "
          f"mean generated tokens {statistics.mean(r['completion_tokens'] for r in reported):.0f}, "
          f"TGC {sum(r['tgc'] for r in reported)}/{len(reported)}")
    stamps = []
    for line in (out / "sidecar.log").read_text().splitlines():
        found = re.match(r"\[sidecar (\S+ \S+) \S+\] trained steps: (\d+)", line)
        if found and int(found.group(2)) >= 1:
            stamps.append(time.mktime(time.strptime(found.group(1), "%Y-%m-%d %H:%M:%S")))
    if len(stamps) >= 2:
        per_step = statistics.median(b - a for a, b in zip(stamps, stamps[1:])) / 60
        print(f"[info] median {per_step:.1f} min/step; 90 steps ~ {90 * per_step / 60:.1f} h")
    report(results)


def report(results: list[tuple[str, bool, str, str]]) -> None:
    for name, ok, expected, actual in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<12} expected {expected}; actual {actual}")
    passed = all(ok for _, ok, _, _ in results)
    print("SMOKE PASSED" if passed else "SMOKE FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
