"""Merge a LoRA adapter into the base weights in bf16, the way PRPO's checkpoints were merged.

PRPO's FinQA scores come from bf16-merged models (gitlab/tail/rllm/finqa-grpo-run/
merge_lora.py: base loaded in bfloat16, PEFT merge_and_unload, save). A bf16 merge
keeps only part of a small LoRA update (measured 2026-09-24 on a DeepCoder SAO
adapter: 5-12% of the delta survived), so the SAO checkpoints are evaluated both
ways: merged here, to compare with PRPO's recorded numbers under the same
protocol, and LoRA-served, for what the adapter actually does.

usage: merge_adapter.py BASE_DIR ADAPTER_DIR OUT_DIR
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json")


def main() -> None:
    base, adapter, out = (Path(arg) for arg in sys.argv[1:4])
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    reference = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith("mlp.down_proj.weight")}
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    changed = [(model.get_parameter(n) != w).float().mean().item() for n, w in reference.items()]
    print(f"merged {adapter}: {100 * sum(changed) / len(changed):.2f}% of down_proj elements changed in bf16")
    model.save_pretrained(out, safe_serialization=True)
    for name in TOKENIZER_FILES:  # serve with the base's own tokenizer and chat template, byte for byte
        if (base / name).exists():
            shutil.copy2(base / name, out / name)


if __name__ == "__main__":
    main()
