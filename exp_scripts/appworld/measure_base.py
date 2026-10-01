"""Measure base Qwen3-4B-Instruct-2507 on AppWorld train90 in code modality, 50 steps.

Before building the AppWorld SAO driver, two things must be known:
  1. whether the base has a learning signal at all in code modality (distribution of the
     corrected partial reward -- the fraction of no_op_fail tests passed -- and official TGC);
  2. how long an episode is: turns, wall time, per-turn prompt tokens, and the length of the
     whole episode assembled as one sequence (last turn's prompt + its completion), which
     decides whether one-sample-per-episode fits and how often an episode would straddle a
     weight publication.

The episode loop is grpo_vanilla's own collector (scripts/collect_appworld_trajectories.py:
run_task), imported, not copied, with these settings overridden:
  - policy: the vLLM server given by --endpoint (Qwen3-4B-Instruct-2507, served on .29)
  - MAX_TOOL_CALLS = 50 (AppWorld's official budget for the ReAct/code agents)
  - --align-harness prompt (grpo_vanilla's SYSTEM_PROMPT + harness observation, as in the
    earlier training line); its whole-episode 24576-token wall is lifted and replaced by a
    4096-token cap per turn, so the natural episode length is what gets measured
  - temperature 1.0, as the earlier AppWorld training line
Every model call's usage, finish reason and latency are recorded per turn.

N workers run in parallel, each with its own `appworld serve environment` on its own port and
a disjoint slice of the tasks. Resumable: a task whose output exists is skipped.

usage: measure_base.py --endpoint URL --model NAME --out DIR [--workers 8] [--port-base 7300]
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

TAIL = Path("/home/yanan/agents/gitlab/tail")
COLLECTOR = TAIL / "grpo_vanilla" / "scripts" / "collect_appworld_trajectories.py"
MAX_STEPS = 50
TURN_MAX_TOKENS = 4096


def load_collector():
    spec = importlib.util.spec_from_file_location("collect_appworld_trajectories", COLLECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task_records() -> dict[str, dict]:
    """Per-task manifest records (required_apis) for function_calling, as the collector's main() loads them."""
    records: dict[str, dict] = {}
    for path in sorted((TAIL / "grpo_vanilla" / "runs").glob("*/manifest/appworld_train90_dev57.json")):
        for record in json.loads(path.read_text())["tasks"]:
            if record.get("required_apis"):
                records.setdefault(record["task_id"], record)
    return records


def auto_print(code: str) -> str:
    """Wrap a trailing bare expression (other than a print call) in print(), as a REPL shows it."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1].value
    if isinstance(last, ast.Call) and isinstance(last.func, ast.Name) and last.func.id == "print":
        return code
    tree.body[-1] = ast.Expr(ast.Call(ast.Name("print", ast.Load()), [last], []))
    return ast.unparse(ast.fix_missing_locations(tree))


def worker(index: int, tasks: list[str], args: argparse.Namespace) -> None:
    c = load_collector()
    c.ENDPOINT = args.endpoint.rstrip("/") + "/chat/completions"
    c.MODEL = args.model
    c.API_KEY = "EMPTY"
    c.MAX_TOOL_CALLS = args.max_steps
    # Episode wall: the harness caps the whole episode's generated tokens (max_completion_length).
    # --episode-token-wall N reproduces it exactly (no per-turn cap, as the harness has none);
    # without it the wall is lifted and each turn is capped at TURN_MAX_TOKENS instead.
    c.HARNESS_COMPLETION_BUDGET = args.episode_token_wall or 10**9
    turn_cap = args.episode_token_wall or TURN_MAX_TOKENS
    turns: list[dict] = []
    original_chat = c.chat

    def chat(messages, tools, temperature, max_tokens=TURN_MAX_TOKENS):
        started = time.monotonic()
        response = original_chat(messages, tools, temperature, max_tokens=min(max_tokens, turn_cap))
        usage = response.get("usage") or {}
        choice = response["choices"][0]
        turns.append({
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "tool_calls": len((choice.get("message") or {}).get("tool_calls") or []),
            "seconds": round(time.monotonic() - started, 2),
        })
        return response

    c.chat = chat
    if args.auto_print_last_expr:
        # Measurement variant, not the earlier harness: show the value of a trailing bare
        # expression the way an IPython/Jupyter cell does, so `apis.api_docs.show_...()`
        # without print() is no longer silent.
        original_post = c.Server.post

        def post(self, path, payload, *a, **k):
            if path == "/execute":
                payload = dict(payload, code=auto_print(payload["code"]))
            return original_post(self, path, payload, *a, **k)

        c.Server.post = post
    out = Path(args.out)
    records = load_task_records() if args.modality == "function_calling" else {}
    for task_id, rnd in tasks:
        target = out / "episodes" / (f"{task_id}.json" if args.rounds == 1 else f"{task_id}__r{rnd:02d}.json")
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        turns.clear()
        server = c.Server(args.port_base + index, Path(args.appworld_root), TAIL / "appworld" / ".venv",
                          out / "servers" / f"server-{index}.log")
        try:
            server.start()
            row = c.run_task(server, task_id, records.get(task_id, {}), args.temperature,
                             f"sao-measure/{Path(args.out).name}/{task_id}/r{rnd:02d}",
                             Path(args.appworld_root), args.modality, True)
            last = turns[-1] if turns else {}
            row.update({
                "turns": list(turns),
                "n_turns": len(turns),
                "assembled_tokens": (last.get("prompt_tokens") or 0) + (last.get("completion_tokens") or 0),
                "sum_prompt_tokens": sum(t["prompt_tokens"] or 0 for t in turns),
                "turns_hit_cap": sum(t["finish_reason"] == "length" for t in turns),
                "round": rnd, "max_steps": args.max_steps, "episode_token_wall": args.episode_token_wall,
                "turn_cap": turn_cap, "modality": args.modality,
            })
            target.write_text(json.dumps(row))
            print(f"[w{index}] {task_id} reward={row['corrected_reward']} tgc={row['official_tgc']} "
                  f"turns={row['n_turns']} calls={row['tool_calls']} stop={row['stop_reason']} "
                  f"assembled={row['assembled_tokens']} {row['elapsed_s']}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - recorded, the sweep continues
            (out / "errors").mkdir(parents=True, exist_ok=True)
            (out / "errors" / f"{task_id}.txt").write_text(f"{type(exc).__name__}: {exc}")
            print(f"[w{index}] {task_id} ERROR {type(exc).__name__}: {exc}", flush=True)
        finally:
            server.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--port-base", type=int, default=7300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--modality", choices=("code", "function_calling"), default="code")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--episode-token-wall", type=int, default=None,
                        help="cap the whole episode's generated tokens, as the harness's max_completion_length")
    parser.add_argument("--rounds", type=int, default=1, help="independent episodes per task")
    parser.add_argument("--appworld-root", default=str(TAIL / "appworld"),
                        help="writable AppWorld root (data/ + experiments outputs); local disk, not NFS. "
                             "The server binary stays in the shared checkout's .venv")
    parser.add_argument("--auto-print-last-expr", action="store_true",
                        help="variant: print a trailing bare expression's value (REPL display)")
    args = parser.parse_args()
    listing = TAIL / "appworld" / "data" / "datasets" / f"{args.split}.txt"
    tasks = [t.strip() for t in listing.read_text().splitlines() if t.strip()]
    if args.task_ids:
        tasks = [t for t in tasks if t in set(args.task_ids.split(","))]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "config.json").write_text(json.dumps({**vars(args), "tasks": len(tasks),
                                                            "turn_max_tokens": TURN_MAX_TOKENS,
                                                            "auth_hint_env": os.environ.get("GRPO_VANILLA_APPWORLD_AUTH_HINT", "1"),
                                                            "react_prompt_env": os.environ.get("GRPO_VANILLA_APPWORLD_REACT_PROMPT", "")}))
    print(f"[measure] {len(tasks)} tasks, {args.workers} workers, endpoint {args.endpoint}", flush=True)
    jobs = [(t, r) for r in range(1, args.rounds + 1) for t in tasks]
    slices = [jobs[i::args.workers] for i in range(args.workers)]
    procs = [multiprocessing.Process(target=worker, args=(i, s, args)) for i, s in enumerate(slices) if s]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("[measure] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
