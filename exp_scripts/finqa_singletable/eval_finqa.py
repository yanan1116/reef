"""Evaluate one served model on FinQA val or test, with the PRPO evaluation protocol.

PRPO's FinQA checkpoints were scored by `rllm eval finqa`
(gitlab/tail/rllm/finqa-grpo-run/eval_parallel.sh): rllm's finqa_flow against a
vLLM server, sampling temperature 0 / top_p 1.0 / seed 1234 and no max_tokens
(the server fills up to max_model_len 12288), a 300 s timeout per model call,
32 tasks in flight, and finqa_evaluator (gpt-5.4-nano judge). This script runs
the same flow (finqa_env.run_episode) and evaluator without rllm.

--attempts k runs every task k times as independent requests; with k > 1 at a
temperature above 0 the seed must be dropped (--seed none), or every attempt
repeats one sample. score is the mean over all k x N rollouts (avg@k, the
unbiased pass@1); pass_at[j] for j <= k is also written.

usage: eval_finqa.py --base-url URL --model NAME --split val|test --output DIR
                     [--temperature 0] [--top-p 1.0] [--seed 1234|none] [--attempts 1] [--concurrency 32]
"""

from __future__ import annotations

import argparse
import json
import random
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge_env import load_judge_env  # noqa: E402

load_judge_env()
import openai  # noqa: E402

from finqa_env import TOOL_SPECS, ChatModel, grade, run_episode  # noqa: E402

CALL_TIMEOUT_S = 300  # finqa_flow's per-call timeout
EXPECTED_TASKS = {"val": 522, "test": 558}


class VllmChat(ChatModel):
    """One FinQA turn against an OpenAI-compatible vLLM server, with the evaluation sampling."""

    def __init__(self, client: openai.OpenAI, model: str, sampling: dict, template_kwargs: dict | None = None) -> None:
        self.client = client
        self.model = model
        self.sampling = sampling
        self.extra_body = {"chat_template_kwargs": template_kwargs} if template_kwargs else None

    def complete(self, messages: list[dict]) -> tuple[dict, dict]:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=TOOL_SPECS, timeout=CALL_TIMEOUT_S,
            extra_body=self.extra_body, **self.sampling
        )
        choice = response.choices[0]
        usage = response.usage
        return choice.message.model_dump(exclude_none=True), {
            "finish_reason": choice.finish_reason,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        }


def pass_at_k(counts: list[tuple[int, int]], k: int) -> float:
    """Unbiased pass@k over (n attempts, c correct) per task (Chen et al., 2021)."""
    values = []
    for n, c in counts:
        values.append(1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k))
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=sorted(EXPECTED_TASKS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", default="1234")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--chat-template-kwargs", default=None,
                        help="JSON, e.g. '{\"enable_thinking\": false}' for Qwen3.5 (as eval_multitable.py)")
    parser.add_argument("--shard", default="0/1",
                        help="i/n: evaluate a random (seed 0) 1/n of the tasks (one GPU of n); merge with merge_eval_shards.py")
    args = parser.parse_args()
    if args.attempts > 1 and args.temperature > 0 and args.seed != "none":
        parser.error("--attempts > 1 at --temperature > 0 needs --seed none, or every attempt repeats one sample")

    tasks_path = HERE / "data" / f"finqa_{args.split}.jsonl"
    rows = [json.loads(line) for line in open(tasks_path) if line.strip()]
    if len(rows) != EXPECTED_TASKS[args.split]:
        raise SystemExit(f"{tasks_path}: {len(rows)} tasks, expected {EXPECTED_TASKS[args.split]}")
    shard_index, shard_count = (int(part) for part in args.shard.split("/"))
    if not 0 <= shard_index < shard_count:
        parser.error(f"--shard {args.shard}: expected i/n with 0 <= i < n")
    # Random but reproducible 1/n of the tasks per GPU: shuffle positions with a fixed seed, deal them out.
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    rows = [rows[position] for position in sorted(order[shard_index::shard_count])]
    sampling: dict = {"temperature": args.temperature, "top_p": args.top_p}
    if args.seed != "none":
        sampling["seed"] = int(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "model": args.model, "split": args.split, "tasks": len(rows), "shard": args.shard, "attempts": args.attempts,
        "sampling": sampling, "max_tokens": None, "call_timeout_s": CALL_TIMEOUT_S, "concurrency": args.concurrency,
    }
    template_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    if template_kwargs:  # absent otherwise, so 2507 protocols stay byte-identical
        protocol["chat_template_kwargs"] = template_kwargs
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=2)
    chat = VllmChat(client, args.model, sampling, template_kwargs)

    def evaluate(job: tuple[int, dict]) -> dict:
        attempt, row = job
        episode = run_episode(chat, row["task"]["question"])
        reward, is_correct, grading = grade(row["task"], episode)
        return {
            "idx": row["problem_idx"], "attempt": attempt, "reward": reward, "is_correct": is_correct,
            "turns": len(episode.turns), "ended": episode.ended,
            "error": f"{type(episode.failure).__name__}: {episode.failure}" if episode.failure else None,
            "table_access": grading.get("table_access"),
        }

    started = time.time()
    jobs = [(attempt, row) for row in rows for attempt in range(args.attempts)]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        items = list(pool.map(evaluate, jobs))
    by_task: dict[int, list[bool]] = {}
    for item in items:
        by_task.setdefault(item["idx"], []).append(item["is_correct"])
    correct = sum(item["is_correct"] for item in items)
    result = {
        "dataset_name": "finqa", "split": args.split, "model": args.model, "total": len(items), "correct": correct,
        "score": correct / len(items), "errors": sum(item["error"] is not None for item in items),
        "attempts": args.attempts, "seconds": round(time.time() - started, 1), "items": items,
    }
    if args.attempts > 1:
        counts = [(len(v), sum(v)) for v in by_task.values()]
        result["pass_at"] = {str(k): pass_at_k(counts, k) for k in range(1, args.attempts + 1)}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[eval] {args.split} COMPLETE correct={correct}/{len(items)} score={result['score']:.6f} errors={result['errors']}", flush=True)


if __name__ == "__main__":
    main()
