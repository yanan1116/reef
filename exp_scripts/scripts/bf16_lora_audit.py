"""Does a SAO LoRA adapter survive bf16? Weight level and output level.

For every adapted module, dW = (alpha/r) * B @ A computed in fp32 from the adapter
as stored, against the bf16 base weight W:

  rel_fro        ||dW||_F / ||W||_F                     size of the update
  merge_kept     ||bf16(W + dW) - W||_F / ||dW||_F      what a bf16 MERGE keeps
  merge_changed  fraction of W elements a bf16 merge changes at all
  serve_kept     ||bf16(y + dy) - bf16(y)|| / ||dy||    what UNMERGED serving keeps:
                 y = x W^T and dy = x dW^T in fp32, rounded to bf16, added in fp32
                 and stored bf16, as vLLM's LoRA expand kernel does. x is Gaussian
                 (64 rows): a proxy for real hidden states, which have outliers.
  serve_cos      cosine(bf16(y+dy) - bf16(y), dy)       direction preserved when serving

usage: bf16_lora_audit.py BASE_DIR NAME=ADAPTER_DIR [NAME=ADAPTER_DIR ...]
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import torch
from safetensors import safe_open

torch.manual_seed(0)
base_dir = Path(sys.argv[1])
index = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
handles = {}
def base_weight(key):
    fn = index[key]
    if fn not in handles:
        handles[fn] = safe_open(str(base_dir / fn), "pt")
    return handles[fn].get_tensor(key)

def audit(adapter_dir):
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    f = safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt")
    acc = defaultdict(lambda: defaultdict(float))
    for ka in sorted(k for k in f.keys() if k.endswith("lora_A.weight")):
        kb = ka.replace("lora_A", "lora_B")
        mod = re.sub(r"^base_model\.model\.", "", ka).replace(".lora_A.weight", ".weight")
        kind = mod.split(".")[-2]
        A, B = f.get_tensor(ka).float(), f.get_tensor(kb).float()
        W = base_weight(mod)
        assert W.dtype == torch.bfloat16, (mod, W.dtype)
        Wf = W.float()
        dW = scale * (B @ A)
        merged = (Wf + dW).to(torch.bfloat16).float()
        x = torch.randn(64, Wf.shape[1])
        y = (x @ Wf.T).to(torch.bfloat16)
        dy = x @ dW.T
        out = (y.float() + dy.to(torch.bfloat16).float()).to(torch.bfloat16)
        got = out.float() - y.float()
        a = acc[kind]
        a["n"] += 1
        a["dW2"] += dW.pow(2).sum().item(); a["W2"] += Wf.pow(2).sum().item()
        a["merge2"] += (merged - Wf).pow(2).sum().item()
        a["changed"] += (merged != Wf).sum().item(); a["elems"] += Wf.numel()
        a["dy2"] += dy.pow(2).sum().item(); a["got2"] += got.pow(2).sum().item()
        a["dot"] += (got * dy).sum().item()
        a["absdW"] += dW.abs().sum().item(); a["absW"] += Wf.abs().sum().item()
    return acc

def summarize(a):
    return dict(
        rel_fro=(a["dW2"] / a["W2"]) ** .5,
        mean_abs_dW=a["absdW"] / a["elems"], mean_abs_W=a["absW"] / a["elems"],
        merge_kept=(a["merge2"] / a["dW2"]) ** .5, merge_changed=a["changed"] / a["elems"],
        serve_kept=(a["got2"] / a["dy2"]) ** .5, serve_cos=a["dot"] / (a["got2"] * a["dy2"]) ** .5,
    )

report = {}
for spec in sys.argv[2:]:
    name, d = spec.split("=", 1)
    acc = audit(Path(d))
    total = defaultdict(float)
    for a in acc.values():
        for k, v in a.items(): total[k] += v
    report[name] = {"ALL": summarize(total), **{k: summarize(v) for k, v in sorted(acc.items())}}
    print(f"done {name}", file=sys.stderr, flush=True)
print(json.dumps(report, indent=1))
