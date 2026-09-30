"""Evaluate one served model on FinQA multi_val or multi_test.

finqa/eval_finqa.py (the single-table SAO evaluator) with the benchmark swapped: the episode is
multitable_env.run_episode, the code SAO trains with (rllm's multitable_v2_flow turn for turn,
final-synthesis calls keeping the tool list), against a vLLM server; the grade is rllm's
finqa_evaluator (gpt-5.4-nano multi-table rubric). Per-call caps are the flow's (2048 tool-use /
8192 final, 300 s timeout); a call vLLM refuses (e.g. prompt + max_tokens > 49152) or that fails
is the flow's failed call (retried once, then the flow moves on).

--attempts k runs every task k times as independent requests; with k > 1 at a temperature above
0 the seed must be dropped (--seed none). Reported: success rate (rubric score >= 0.9, the
evaluator's is_correct) and average rubric score, both over all k x N rollouts (avg@k).

usage: eval_multitable.py --base-url URL --model NAME --split multi_val|multi_test --output DIR
                          [--temperature 0] [--top-p 1.0] [--seed 1234|none] [--attempts 1] [--concurrency 32]
                          [--chat-template-kwargs JSON]   (e.g. '{"enable_thinking": false}' for Qwen3.5)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "finqa_singletable"))
from judge_env import load_judge_env  # noqa: E402

os.environ.setdefault("FINQA_MULTI_TABLE_JUDGE_MODEL", "gpt-5.4-nano")  # before judge_env's default (mini)
load_judge_env()
import openai  # noqa: E402

from multitable_env import TOOL_SPECS, ChatModel, ModelCallRejected, flow, grade, run_episode  # noqa: E402

EXPECTED_TASKS = {"multi_val": 126, "multi_test": 131}


class VllmChat(ChatModel):
    """One multi-table call against an OpenAI-compatible vLLM server, with the evaluation sampling."""

    def __init__(self, client: openai.OpenAI, model: str, sampling: dict, template_kwargs: dict | None) -> None:
        self.client = client
        self.model = model
        self.sampling = sampling
        self.extra_body = {"chat_template_kwargs": template_kwargs} if template_kwargs else None

    def complete(self, messages: list[dict], max_tokens: int) -> tuple[dict, dict]:
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOL_SPECS, max_completion_tokens=max_tokens,
                timeout=flow.LLM_TIMEOUT_SECONDS, extra_body=self.extra_body, **self.sampling,
            )
        except openai.OpenAIError as error:  # the flow's _create_with_retry catches every call failure
            raise ModelCallRejected(f"{type(error).__name__}: {error}") from error
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
    parser.add_argument("--chat-template-kwargs", default=None)
    args = parser.parse_args()
    template_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    if args.attempts > 1 and args.temperature > 0 and args.seed != "none":
        parser.error("--attempts > 1 at --temperature > 0 needs --seed none, or every attempt repeats one sample")

    tasks_path = HERE / "data" / f"{args.split}.jsonl"
    rows = [json.loads(line) for line in open(tasks_path) if line.strip()]
    if len(rows) != EXPECTED_TASKS[args.split]:
        raise SystemExit(f"{tasks_path}: {len(rows)} tasks, expected {EXPECTED_TASKS[args.split]}")
    sampling: dict = {"temperature": args.temperature, "top_p": args.top_p}
    if args.seed != "none":
        sampling["seed"] = int(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "model": args.model, "split": args.split, "tasks": len(rows), "attempts": args.attempts, "sampling": sampling,
        "max_tokens": {"tool_use": flow.DISCOVERY_MAX_COMPLETION_TOKENS, "final": flow.FINAL_MAX_COMPLETION_TOKENS},
        "call_timeout_s": flow.LLM_TIMEOUT_SECONDS, "concurrency": args.concurrency,
        "judge_model": os.environ["FINQA_MULTI_TABLE_JUDGE_MODEL"], "protocol_version": flow.PROTOCOL_VERSION,
        "final_calls_keep_tools": True, "chat_template_kwargs": template_kwargs,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=2)  # the SDK default the flow ran with
    chat = VllmChat(client, args.model, sampling, template_kwargs)

    def evaluate(job: tuple[int, dict]) -> dict:
        attempt, row = job
        episode = run_episode(chat, row["task"])
        reward, is_correct, grading = grade(row["task"], episode)
        return {
            "idx": row["problem_idx"], "attempt": attempt, "reward": reward, "is_correct": is_correct,
            "turns": len(episode.turns), "tool_calls": episode.tool_calls, "ended": episode.finalize_reason,
            "final_fallback_used": episode.final_fallback_used, "llm_errors": episode.llm_errors,
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
        "dataset_name": "finqa_multitable", "split": args.split, "model": args.model, "total": len(items),
        "correct": correct, "score": correct / len(items),
        "mean_reward": sum(item["reward"] for item in items) / len(items),
        "errors": sum(bool(item["llm_errors"]) for item in items),
        "attempts": args.attempts, "seconds": round(time.time() - started, 1), "items": items,
    }
    if args.attempts > 1:
        counts = [(len(v), sum(v)) for v in by_task.values()]
        result["pass_at"] = {str(k): pass_at_k(counts, k) for k in range(1, args.attempts + 1)}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[eval] {args.split} COMPLETE correct={correct}/{len(items)} score={result['score']:.6f} "
          f"mean_reward={result['mean_reward']:.6f} errors={result['errors']}", flush=True)


if __name__ == "__main__":
    main()
