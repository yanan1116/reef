"""Join measure_base.py shards into one AppWorld result: mean corrected reward, TGC and SGC.

usage: summarize_appworld.py OUT_DIR SPLIT_LIST SHARD_DIR [SHARD_DIR ...]
TGC = fraction of tasks whose official tests all pass; SGC = fraction of scenarios (task id before the
"_N" variant suffix) whose variants all pass. Fails unless the shards cover the split exactly once.
"""
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    out, split_list, shard_dirs = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
    expected = split_list.read_text().split()
    episodes = {}
    for shard in shard_dirs:
        for path in sorted((shard / "episodes").glob("*.json")):
            row = json.loads(path.read_text())
            if row["task_id"] in episodes:
                raise SystemExit(f"task {row['task_id']} appears in two shards")
            episodes[row["task_id"]] = row
    missing = sorted(set(expected) - set(episodes))
    if missing or len(episodes) != len(expected):
        raise SystemExit(f"shards cover {len(episodes)} of {len(expected)} tasks; missing {missing[:10]}")
    rewards = {t: float(r["corrected_reward"]) for t, r in episodes.items()}
    tgc = {t: bool(r["official_tgc"]) for t, r in episodes.items()}
    unscored = sorted(t for t, r in episodes.items() if r.get("verifier_status") != "scored")
    scenarios = defaultdict(list)
    for task in expected:
        scenarios[task.rsplit("_", 1)[0]].append(tgc[task])
    n = len(expected)
    summary = {
        "tasks": n,
        "mean_corrected_reward": sum(rewards.values()) / n,
        "nonzero_reward": sum(v > 0 for v in rewards.values()),
        "tgc": sum(tgc.values()) / n, "tgc_count": sum(tgc.values()),
        "sgc": sum(all(v) for v in scenarios.values()) / len(scenarios),
        "sgc_count": sum(all(v) for v in scenarios.values()), "scenarios": len(scenarios),
        "unscored": unscored,
        "stop_reasons": dict(sorted({s: sum(r["stop_reason"] == s for r in episodes.values())
                                     for s in {r["stop_reason"] for r in episodes.values()}}.items())),
        "mean_completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in episodes.values()) / n,
        "config": json.loads((shard_dirs[0] / "config.json").read_text()) | {"task_ids": "per shard", "endpoint": "per shard"},
        "host": ".36 Quadro RTX 5000 (sm_75)", "dtype": "float16", "vllm": "0.22.1+cu129",
        "attention_backend": "TRITON_ATTN", "gpu_groups": os.environ.get("GPU_GROUPS_USED"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    episodes_out = out / "episodes"
    episodes_out.mkdir(exist_ok=True)
    for shard in shard_dirs:
        for path in (shard / "episodes").glob("*.json"):
            shutil.copy2(path, episodes_out / path.name)
    print(f"[appworld] {n} tasks: mean reward {summary['mean_corrected_reward']:.4f}, "
          f"TGC {summary['tgc_count']}/{n} = {summary['tgc']:.4f}, SGC {summary['sgc_count']}/{len(scenarios)} = {summary['sgc']:.4f}, "
          f"unscored {len(unscored)}")


if __name__ == "__main__":
    main()
