"""Item-level comparison of the lr5x SAO evals against the base runs, .29 results only
(SAO checkpoints vs base, both served by the same LoRA-enabled vLLM setup)."""
import itertools
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results" / "deepcoder"
B29 = Path("/mnt/disk1t/deepcoder-prpo-checkpoint-eval")
runs = {}
for step in (20, 40, 60, 80, 100, 120, 140, 160, 180, 190):
    for p in sorted((R / "eval-c32-lora-lr5x").glob(f"lr5x_step_{step:03d}*/result.json")):
        suffix = p.parent.name[len(f"lr5x_step_{step:03d}"):].lstrip("_")
        runs[f"lr5x step {step} (.29){' ' + suffix if suffix else ''}"] = p
runs["Base (.29) r1"] = R / "eval-c32-lora-lr5x/base_29/result.json"
for rep in sorted((R / "eval-c32-lora-lr5x").glob("base_29_r*/result.json")):
    runs[f"Base (.29) {rep.parent.name[-2:]}"] = rep

res = {}
for name, path in runs.items():
    if not path.exists():
        continue
    d = json.loads(path.read_text())
    assert d["total"] == 687, (name, d["total"])
    res[name] = ({i["idx"]: bool(i["is_correct"]) for i in d["items"]}, d["correct"], d["errors"])

w = max(len(n) for n in res)
print(f"| {'Run':<{w}} | Success Rate     | Avg Score | Errors |")
print(f"|{'-' * (w + 2)}|------------------|-----------|--------|")
for n, (_, c, e) in res.items():
    print(f"| {n:<{w}} | {100 * c / 687:5.2f}% ({c}/687) | {c / 687:.3f}     | {e:<6} |")

bases = [n for n in res if n.startswith("Base")]
models = [n for n in res if not n.startswith("Base")]


def stats(pairs):
    s = [sum(res[a][0][k] and not res[b][0][k] for k in res[a][0]) for a, b in pairs]
    br = [sum(res[b][0][k] and not res[a][0][k] for k in res[a][0]) for a, b in pairs]
    nets = [x - y for x, y in zip(s, br)]
    return (str(len(pairs)), f"{min(s)}-{max(s)}", f"{min(br)}-{max(br)}", f"{min(nets):+d} to {max(nets):+d}")


rows = [(f"{m} vs bases", *stats([(m, b) for b in bases])) for m in models]
rows.append(("Base vs base (noise)", *stats(list(itertools.combinations(bases, 2)))))
head = ("Comparison", "Pairs", "Newly Solved", "Newly Broken", "Net range")
w = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
print()
print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(head)) + " |")
print("|" + "|".join("-" * (x + 2) for x in w) + "|")
for r in rows:
    print("| " + " | ".join(v.ljust(w[i]) for i, v in enumerate(r)) + " |")
