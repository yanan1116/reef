"""Does a kept LoRA adapter reproduce the policy that generated the training rollouts?

For turns sampled by weight version V (extract_rollout_turns.py: tokens + the engine's per-token
rollout log-probs), recompute the response tokens' log-probs with the base model plus each
candidate adapter (the HF PEFT files the evaluation mounts in vLLM), and with the base alone,
on CPU in fp32. The adapter saved for V must be the closest to the engine's log-probs, clearly
closer than the base; the gap left over is bf16-engine vs fp32 numerics.

Log-probs are compared both raw and with logits / T (T = the rollout temperature), since the
engine may report either; the report says which one the engine's numbers follow. Most response
tokens are near-certain (log-prob ~ 0 under any of the models), so the pooled comparison is also
given on the uncertain tokens only (engine log-prob < -0.1), where the policies actually differ.
Per-token values are written to OUT_JSON for further analysis.

usage: adapter_fidelity.py TURNS_JSON OUT_JSON BASE_DIR TEMPERATURE VERSION=ADAPTER_DIR [VERSION=ADAPTER_DIR ...]
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def response_logprobs(model, tokens: list[int], prompt_length: int, temperature: float) -> tuple[list[float], list[float]]:
    ids = torch.tensor([tokens])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, prompt_length - 1:-1].float()
    target = ids[0, prompt_length:]
    raw = torch.log_softmax(logits, -1).gather(-1, target[:, None])[:, 0]
    scaled = torch.log_softmax(logits / temperature, -1).gather(-1, target[:, None])[:, 0]
    return raw.tolist(), scaled.tolist()


def main() -> None:
    turns_by_version = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
    out_path, base_dir, temperature = sys.argv[2], sys.argv[3], float(sys.argv[4])
    adapters = {int(k): v for k, v in (arg.split("=", 1) for arg in sys.argv[5:])}
    saved: list[dict] = []
    torch.set_num_threads(max(1, torch.get_num_threads()))
    base = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32)
    base.eval()
    names = {v: f"v{v}" for v in adapters}
    first, *rest = sorted(adapters)
    model = PeftModel.from_pretrained(base, adapters[first], adapter_name=names[first])
    for v in rest:
        model.load_adapter(adapters[v], adapter_name=names[v])
    model.eval()

    print(f"{'turn version':>12} {'candidate':>12} {'tokens':>6} {'mean|d| raw':>11} {'mean|d| /T':>10} {'max|d| /T':>9}")
    summary: dict[tuple[int, str], list[float]] = {}
    for version, turns in sorted(turns_by_version.items()):
        near = min(adapters, key=lambda a: abs(a - version))
        for turn in turns:
            engine = turn["rollout_log_probs"]
            for label in ("base", f"adapter@{near}"):
                started = time.time()
                if label == "base":
                    with model.disable_adapter():
                        raw, scaled = response_logprobs(model, turn["tokens"], turn["prompt_length"], temperature)
                else:
                    model.set_adapter(names[near])
                    raw, scaled = response_logprobs(model, turn["tokens"], turn["prompt_length"], temperature)
                saved.append({"version": version, "candidate": label, "agent_record_id": turn["agent_record_id"],
                              "engine": engine, "raw": raw, "scaled": scaled})
                d_raw = [abs(a - b) for a, b in zip(raw, engine)]
                d_scaled = [abs(a - b) for a, b in zip(scaled, engine)]
                summary.setdefault((version, label), []).extend(d_scaled)
                summary.setdefault((version, label + " raw"), []).extend(d_raw)
                print(f"{version:>12} {label:>12} {len(engine):>6} {statistics.mean(d_raw):>11.4f} "
                      f"{statistics.mean(d_scaled):>10.4f} {max(d_scaled):>9.3f}   ({time.time() - started:.0f}s)", flush=True)
    json.dump(saved, open(out_path, "w"))
    print()
    print("uncertain tokens only (engine log-prob < -0.1), pooled per version: mean |d| (logits / T | raw), n tokens")
    for version in sorted(turns_by_version):
        for label in sorted({s["candidate"] for s in saved if s["version"] == version}):
            ds, dr = [], []
            for s in saved:
                if s["version"] == version and s["candidate"] == label:
                    for e, a, r in zip(s["engine"], s["scaled"], s["raw"]):
                        if e < -0.1:
                            ds.append(abs(a - e)); dr.append(abs(r - e))
            if ds:
                print(f"  version {version} {label:>12}: {statistics.mean(ds):.4f} | {statistics.mean(dr):.4f}  n={len(ds)}")
    print()
    print("per version, all sampled response tokens pooled (logits / T):")
    for version in sorted(turns_by_version):
        cells = {k[1]: statistics.mean(v) for k, v in summary.items() if k[0] == version and not k[1].endswith(" raw")}
        raws = {k[1][:-4]: statistics.mean(v) for k, v in summary.items() if k[0] == version and k[1].endswith(" raw")}
        print(f"  version {version}: " + ", ".join(f"{k} {cells[k]:.4f} (raw {raws[k]:.4f})" for k in sorted(cells)))


if __name__ == "__main__":
    main()
