"""Which part / scale of a kept LoRA adapter reproduces the engine's rollout log-probs?

adapter_fidelity.py found the full adapter far from the engine's log-probs while the base is
close. This scans variants of the adapter on the same turns: the whole delta scaled by s, and the
delta restricted to module groups (attention q/k/v, o_proj, MLP gate/up, down_proj). Metric: mean
|log-prob - engine log-prob| (logits / T) on uncertain tokens (engine log-prob < -0.1) and on all
response tokens. A variant that lands clearly below the base locates the correct part of the
export; if nothing beats the base, the policy the engine served is not in this file at any scale.

usage: adapter_scale_scan.py TURNS_JSON BASE_DIR TEMPERATURE VERSION=ADAPTER_DIR [...]
"""

from __future__ import annotations

import json
import statistics
import sys

import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM

GROUPS = {
    "all": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "o": ("o_proj",),
    "gate_up": ("gate_proj", "up_proj"),
    "down": ("down_proj",),
}
VARIANTS = [("base", "all", 0.0)] + [(f"all x{s}", "all", s) for s in (0.1, 0.25, 0.5, 1.0)] + \
           [(f"{g} only", g, 1.0) for g in ("qkv", "o", "gate_up", "down")]


def main() -> None:
    turns_by_version = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
    base_dir, temperature = sys.argv[2], float(sys.argv[3])
    adapters = {int(k): v for k, v in (arg.split("=", 1) for arg in sys.argv[4:])}
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32).eval()
    first, *rest = sorted(adapters)
    model = PeftModel.from_pretrained(base, adapters[first], adapter_name=f"v{first}")
    for v in rest:
        model.load_adapter(adapters[v], adapter_name=f"v{v}")
    model.eval()
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    original = {(name, a): module.scaling[a] for name, module in layers for a in module.scaling}

    def configure(adapter: str, group: str, scale: float) -> None:
        for name, module in layers:
            in_group = name.split(".")[-1] in GROUPS[group]
            module.scaling[adapter] = original[(name, adapter)] * (scale if in_group else 0.0)

    print(f"{'version':>7} {'variant':>14} {'uncertain':>9} {'all':>7}  (mean |d|, logits / T)")
    for version, turns in sorted(turns_by_version.items()):
        if version not in adapters:
            continue
        adapter = f"v{version}"
        model.set_adapter(adapter)
        turns = [t for t in turns if any(e < -0.1 for e in t["rollout_log_probs"])]
        if not turns:
            continue
        for label, group, scale in VARIANTS:
            configure(adapter, group, scale)
            unc, every = [], []
            for turn in turns:
                ids = torch.tensor([turn["tokens"]])
                with torch.no_grad():
                    logits = model(input_ids=ids).logits[0, turn["prompt_length"] - 1:-1].float() / temperature
                lp = torch.log_softmax(logits, -1).gather(-1, ids[0, turn["prompt_length"]:, None])[:, 0].tolist()
                for a, e in zip(lp, turn["rollout_log_probs"]):
                    every.append(abs(a - e))
                    if e < -0.1:
                        unc.append(abs(a - e))
            print(f"{version:>7} {label:>14} {statistics.mean(unc):>9.4f} {statistics.mean(every):>7.4f}   n={len(unc)}/{len(every)}", flush=True)
        configure(adapter, "all", 1.0)


if __name__ == "__main__":
    main()
