"""Measure the observation cross-entropy ECHO sees, with vLLM prompt log-probs (user, 2026-10-23: is echo_ce ~7 real?).

Renders AppWorld episodes from a driver's episodes.jsonl with the Qwen3.5 chat template, scores every token with the
base model's prompt log-probs, and reports the mean -log p over (a) ECHO's tokens (recipes/sao/slime/echo.py masks
with the given rules), (b) the model's own action tokens, (c) all tokens after the first prompt.

    python echo_ce_check.py MODEL_DIR EPISODES_JSONL RULES_JSON [N_EPISODES=5]
"""

import importlib.util
import json
import statistics
import sys
import types

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def main() -> None:
    model_dir, episodes_path, rules_path = sys.argv[1:4]
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    # echo.py imports ECHO_SPANS from the SAO package, which needs Reef; load the file alone with that one name.
    stub = types.ModuleType("recipes.sao.slime")
    stub.ECHO_SPANS = ("tool", "all")
    for name in ("recipes", "recipes.sao"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["recipes.sao.slime"] = stub
    spec = importlib.util.spec_from_file_location("echo", "/home/yanan/agents/reef/recipes/sao/slime/echo.py")
    echo = importlib.util.module_from_spec(spec)
    sys.modules["echo"] = echo
    spec.loader.exec_module(echo)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rules = echo.EchoRules.load(rules_path)
    open_id, close_id = tokenizer.convert_tokens_to_ids("<tool_response>"), tokenizer.convert_tokens_to_ids("</tool_response>")
    im_start, im_end = tokenizer.convert_tokens_to_ids("<|im_start|>"), tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant = tokenizer.encode("assistant", add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    tools = [{"type": "function", "function": {"name": "execute_code", "description": "Run Python code",
              "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]

    samples = []
    for line in open(episodes_path):
        episode = json.loads(line)
        messages = episode["messages"]
        for message in messages:
            for call in message.get("tool_calls") or []:
                arguments = call["function"].get("arguments")
                call["function"]["arguments"] = json.loads(arguments) if isinstance(arguments, str) else arguments
        ids = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=False,
                                            enable_thinking=False)
        ids = ids if isinstance(ids, list) else ids["input_ids"]
        if len(ids) > 32000:
            continue
        starts = [i for i, token in enumerate(ids) if token == im_start]
        # actions: tokens inside assistant turns after their "assistant\n" header, up to <|im_end|>
        action = [0] * len(ids)
        for start in starts:
            if ids[start + 1 : start + 1 + len(assistant)] == assistant:
                i = start + 1 + len(assistant) + 1
                while i < len(ids) and ids[i] != im_end:
                    action[i] = 1
                    i += 1
        first = next(s for s in starts if ids[s + 1 : s + 1 + len(assistant)] == assistant)
        samples.append((episode.get("task_id"), ids, action, first))
        if len(samples) >= count:
            break

    llm = LLM(model=model_dir, max_model_len=32768, gpu_memory_utilization=0.5, enforce_eager=True, max_num_seqs=1,
              max_num_batched_tokens=2048, enable_chunked_prefill=True,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outputs = llm.generate([{"prompt_token_ids": ids} for _, ids, _, _ in samples],
                           SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0))
    rows = []
    for (task, ids, action, first), output in zip(samples, outputs):
        logprobs = [None] + [entry[token].logprob for entry, token in zip(output.prompt_logprobs[1:], ids[1:])]
        response = len(ids) - first
        action_mask = torch.tensor(action[first:])
        (mask,) = echo.observation_masks([torch.tensor(ids)], [response], [action_mask], "tool", open_id, close_id,
                                         decode, rules)
        lp = torch.tensor(logprobs[first:])
        echo_ce = (-(lp * mask).sum() / mask.sum()).item()
        action_ce = (-(lp * action_mask).sum() / action_mask.sum()).item()
        rows.append((task, len(ids), int(mask.sum()), echo_ce, action_ce, (-lp).mean().item()))
        if len(rows) == 1:  # the first episode's ECHO tokens, worst first and in order, to see what is hard
            positions = [i for i in range(response) if mask[i] > 0]
            print("first ECHO tokens:", " ".join(f"{decode([ids[first + i]])!r}:{-lp[i].item():.1f}" for i in positions[:80]))
            worst = sorted(positions, key=lambda i: lp[i].item())[:25]
            print("worst ECHO tokens:", " ".join(f"{decode([ids[first + i]])!r}:{-lp[i].item():.1f}" for i in worst))
            share = sum(1 for i in positions if -lp[i].item() > 10) / len(positions)
            print("share of ECHO tokens with -log p > 10 nats: %.2f" % share)
        print(f"{task}: tokens {len(ids)}, ECHO tokens {int(mask.sum())}, CE echo {echo_ce:.3f} action {action_ce:.3f} "
              f"all-response {(-lp).mean().item():.3f}", flush=True)
    print("mean over episodes: CE echo %.3f, action %.3f" % (statistics.mean(r[3] for r in rows),
                                                             statistics.mean(r[4] for r in rows)))


if __name__ == "__main__":
    main()
