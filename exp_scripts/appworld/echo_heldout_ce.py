"""Held-out observation cross-entropy of SAO / ECHO checkpoints (user, 2026-10-24: diagnose before the next ECHO run).

Question: does ECHO teach a world model that transfers, or does it memorize the 90 training tasks' tool output? For
every model (the base and each LoRA adapter) this scores AppWorld episodes with vLLM prompt log-probs and reports the
mean -log p per episode on ECHO's tokens under each rules file (recipes/sao/slime/echo.py masks, span "tool"), and on
the policy's own action tokens. Episode sets: held-out dev episodes and training-task episodes, both played by the
base model, so every checkpoint is scored on the same text.

    python echo_heldout_ce.py --model MODEL_DIR --out OUT.jsonl \
        --episodes dev=DEV.jsonl --episodes train=TRAIN.jsonl \
        --rules paper=echo_rules.json --rules nodocs=echo_rules_no_api_docs.json \
        --adapter echo@30=ADAPTER_DIR ... [--max-episodes 40]

One JSON line per (model, episode set, episode) goes to OUT; a summary table is printed at the end.
"""

import argparse
import importlib.util
import json
import statistics
import sys
import types
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MAX_TOKENS = 32000


def load_echo_module():
    # echo.py imports ECHO_SPANS from the SAO package, which needs Reef; load the file alone with that one name.
    stub = types.ModuleType("recipes.sao.slime")
    stub.ECHO_SPANS = ("tool", "all")
    for name in ("recipes", "recipes.sao"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["recipes.sao.slime"] = stub
    spec = importlib.util.spec_from_file_location("echo", Path(__file__).resolve().parents[2] / "recipes/sao/slime/echo.py")
    echo = importlib.util.module_from_spec(spec)
    sys.modules["echo"] = echo
    spec.loader.exec_module(echo)
    return echo


def pairs(values: list[str]) -> list[tuple[str, str]]:
    return [tuple(value.split("=", 1)) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", action="append", required=True, help="NAME=episodes.jsonl")
    parser.add_argument("--rules", action="append", required=True, help="NAME=rules.json")
    parser.add_argument("--adapter", action="append", default=[], help="NAME=adapter dir (the base is always scored)")
    parser.add_argument("--max-episodes", type=int, default=40)
    # prompt log-probs materialize full-vocabulary fp32 logits per prefill chunk (248k x chunk x 4 bytes): keep the
    # chunk small and leave room outside the KV cache (8192 tokens OOMed a 32 GB card, 2026-10-24).
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.6)
    args = parser.parse_args()

    echo = load_echo_module()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    open_id = tokenizer.convert_tokens_to_ids("<tool_response>")
    close_id = tokenizer.convert_tokens_to_ids("</tool_response>")
    im_start, im_end = tokenizer.convert_tokens_to_ids("<|im_start|>"), tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant = tokenizer.encode("assistant", add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)  # noqa: E731
    tools = [{"type": "function", "function": {"name": "execute_code", "description": "Run Python code",
              "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
    rules = {name: echo.EchoRules.load(path) for name, path in pairs(args.rules)}

    # Render each episode once; its masks do not depend on the model.
    samples = []  # (set, task, ids, first, action mask, {rules name: mask})
    for set_name, path in pairs(args.episodes):
        taken = 0
        for line in open(path):
            if taken >= args.max_episodes:
                break
            episode = json.loads(line)
            messages = episode["messages"]
            for message in messages:
                for call in message.get("tool_calls") or []:
                    arguments = call["function"].get("arguments")
                    call["function"]["arguments"] = json.loads(arguments) if isinstance(arguments, str) else arguments
            ids = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=False,
                                                enable_thinking=False)
            ids = ids if isinstance(ids, list) else ids["input_ids"]
            if len(ids) > MAX_TOKENS:
                continue
            starts = [i for i, token in enumerate(ids) if token == im_start]
            heads = [s for s in starts if ids[s + 1 : s + 1 + len(assistant)] == assistant]
            if not heads:
                continue
            action = [0] * len(ids)
            for start in heads:
                i = start + 1 + len(assistant) + 1
                while i < len(ids) and ids[i] != im_end:
                    action[i] = 1
                    i += 1
            first = heads[0]
            action_mask = torch.tensor(action[first:])
            masks = {}
            for name, rule in rules.items():
                (mask,) = echo.observation_masks([torch.tensor(ids)], [len(ids) - first], [action_mask], "tool",
                                                 open_id, close_id, decode, rule)
                masks[name] = mask
            samples.append((set_name, episode.get("task_id"), ids, first, action_mask, masks))
            taken += 1
    print(f"{len(samples)} episodes rendered", flush=True)

    adapters = [("base", None)] + pairs(args.adapter)
    llm = LLM(model=args.model, max_model_len=32768, gpu_memory_utilization=args.gpu_mem, enforce_eager=True,
              max_num_seqs=4, max_num_batched_tokens=args.chunk_tokens, enable_chunked_prefill=True, enable_lora=len(adapters) > 1,
              max_lora_rank=32, max_loras=1, limit_mm_per_prompt={"image": 0, "video": 0})
    params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0)
    summary = {}
    with open(args.out, "a") as out:
        for index, (model_name, adapter) in enumerate(adapters):
            lora = LoRARequest(model_name, index, adapter) if adapter else None
            outputs = llm.generate([{"prompt_token_ids": s[2]} for s in samples], params, lora_request=lora)
            for (set_name, task, ids, first, action_mask, masks), output in zip(samples, outputs, strict=True):
                logprobs = torch.tensor([entry[token].logprob for entry, token in
                                         zip(output.prompt_logprobs[first:], ids[first:], strict=True)])
                row = {"model": model_name, "set": set_name, "task_id": task, "tokens": len(ids),
                       "action_ce": (-(logprobs * action_mask).sum() / action_mask.sum()).item()}
                for name, mask in masks.items():
                    count = int(mask.sum())
                    row[f"{name}_tokens"] = count
                    row[f"{name}_ce"] = (-(logprobs * mask).sum() / count).item() if count else None
                out.write(json.dumps(row) + "\n")
                out.flush()
                summary.setdefault((model_name, set_name), []).append(row)
            print(f"scored {model_name}", flush=True)

    names = list(rules) + ["action"]
    print("model".ljust(16), "set".ljust(6), " ".join(f"{name + ' CE':>12}" for name in names), "  episodes")
    for (model_name, set_name), rows in summary.items():
        cells = []
        for name in names:
            values = [r[f"{name}_ce"] for r in rows if r.get(f"{name}_ce") is not None]
            cells.append(f"{statistics.mean(values):12.3f}" if values else f"{'-':>12}")
        print(model_name.ljust(16), set_name.ljust(6), " ".join(cells), f"  {len(rows)}")


if __name__ == "__main__":
    main()
