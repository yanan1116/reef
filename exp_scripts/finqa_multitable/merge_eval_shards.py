"""Merge the per-GPU shards of eval_finqa.py / eval_multitable.py (--shard i/n) into one result.json.

usage: merge_eval_shards.py OUT_DIR SHARD_DIR [SHARD_DIR ...]
Checks that the shards share one protocol (apart from the shard) and together cover the split once.
"""
import json
import math
import sys
from pathlib import Path

EXPECTED_TASKS = {"val": 522, "test": 558, "multi_val": 126, "multi_test": 131}


def pass_at_k(counts, k):
    return sum(1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k) for n, c in counts) / len(counts)


def main() -> None:
    out, shard_dirs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    protocols = [json.loads((d / "protocol.json").read_text()) for d in shard_dirs]
    results = [json.loads((d / "result.json").read_text()) for d in shard_dirs]
    strip = lambda p: {k: v for k, v in p.items() if k not in ("shard", "tasks")}
    if any(strip(p) != strip(protocols[0]) for p in protocols):
        raise SystemExit("shards were run with different protocols")
    split, attempts = protocols[0]["split"], protocols[0]["attempts"]
    items = [item for r in results for item in r["items"]]
    tasks = {item["idx"] for item in items}
    if len(tasks) != EXPECTED_TASKS[split] or len(items) != EXPECTED_TASKS[split] * attempts:
        raise SystemExit(f"shards cover {len(tasks)} tasks / {len(items)} items; expected {EXPECTED_TASKS[split]} x {attempts}")
    correct = sum(bool(item["is_correct"]) for item in items)
    result = {key: results[0][key] for key in ("dataset_name", "split", "model", "attempts")}
    result.update(total=len(items), correct=correct, score=correct / len(items),
                  errors=sum(r["errors"] for r in results), seconds=max(r["seconds"] for r in results),
                  shards=[str(d) for d in shard_dirs])
    if "mean_reward" in results[0]:
        result["mean_reward"] = sum(item["reward"] for item in items) / len(items)
    if attempts > 1:
        by_task = {}
        for item in items:
            by_task.setdefault(item["idx"], []).append(bool(item["is_correct"]))
        result["pass_at"] = {str(k): pass_at_k([(len(v), sum(v)) for v in by_task.values()], k) for k in range(1, attempts + 1)}
    result["items"] = items
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps({**strip(protocols[0]), "tasks": EXPECTED_TASKS[split], "shards": len(shard_dirs)}, indent=2))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[merge] {split}: correct={correct}/{len(items)} score={result['score']:.4f}"
          + (f" mean_reward={result['mean_reward']:.4f}" if "mean_reward" in result else ""))


if __name__ == "__main__":
    main()
