"""Judge noise of the multi-table rubric: grade the same final answer several times per judge model.

Takes episodes of the running multi-table training runs (their final answers are in Reef's agent
records), stratified by the training score, and re-grades each answer REPEATS times with each judge
model through rllm's own finqa_eval._call_judge (same rubric prompt, reasoning effort, schema and
retries as training). The per-answer spread across repeats is the judge's noise; comparing it with
the within-task variance of the base model's avg@4 evaluation (same task, different answers) tells
how much of the reward signal a policy could learn from is grading noise.

Run on .16 (the Reef APIs listen on its 127.0.0.1):
  .venv-finqa/bin/python finqa_multitable/judge_noise.py --out results/finqa_multitable/judge_noise/<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "finqa_singletable"))
from judge_env import load_judge_env  # noqa: E402

os.environ.setdefault("FINQA_MULTI_TABLE_JUDGE_MODEL", "gpt-5.4-nano")
load_judge_env()
import finqa_env  # noqa: E402,F401  puts rllm_finqa (finqa_eval) on sys.path
import finqa_eval  # noqa: E402

RUNS = {  # result dir -> Reef service on .16
    "sao": ("/home/yanan/agents/reef/exp_scripts/results/finqa_multitable", "LATEST_FORMAL_TAG", "http://127.0.0.1:8900"),
    "pvf": ("/home/yanan/agents/reef-pvf/exp_scripts/results/finqa_multitable", "LATEST_PVF_TAG", "http://127.0.0.1:8901"),
}
SCENARIO = "sao-finqa-multitable"
TOKEN = "reef-local"
model_lock = threading.Lock()


def final_answer(service: str, record_id: str) -> str:
    request = urllib.request.Request(
        f"{service}/reef/scenarios/{SCENARIO}/records/{record_id}", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read())
    return body["payload"]["response"]["choices"][0]["message"]["content"] or ""


def pick_episodes(per_bin: int, seed: int) -> list[dict]:
    """Up to ``per_bin`` reported episodes per 0.1 training-score bin, from both runs; answer-bearing endings only."""
    pool = []
    for arm, (root, pointer, service) in RUNS.items():
        tag = (Path(root) / pointer).read_text().strip()
        for line in open(Path(root) / tag / "records.jsonl"):
            record = json.loads(line)
            if record.get("dropped") or record["ended"] not in ("model_final", "training_budget"):
                continue
            pool.append({"arm": arm, "service": service, "run": tag, **record})
    bins: dict[int, list[dict]] = defaultdict(list)
    for record in pool:
        bins[min(9, int(record["score"] * 10))].append(record)
    rng = random.Random(seed)
    chosen = []
    for b in sorted(bins):
        rng.shuffle(bins[b])
        chosen += bins[b][:per_bin]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-bin", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--models", default="gpt-5.4-nano,gpt-5.4-mini")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = {}
    for line in open(HERE / "data" / "multi_train.jsonl"):
        row = json.loads(line)
        tasks[str(row["problem_idx"])] = row["task"]

    answers_path = out / "answers.jsonl"
    if answers_path.exists():
        answers = [json.loads(line) for line in open(answers_path)]
    else:
        answers = []
        for record in pick_episodes(args.per_bin, args.seed):
            text = final_answer(record["service"], record["agent_record_ids"][-1])
            if not text.strip():
                continue
            answers.append({"key": f"{record['arm']}:{record['position']}", "arm": record["arm"], "run": record["run"],
                            "problem_idx": str(record["problem_idx"]), "train_score": record["score"],
                            "ended": record["ended"], "answer": text})
        with open(answers_path, "w") as handle:
            for answer in answers:
                handle.write(json.dumps(answer) + "\n")
    print(f"{len(answers)} answers", flush=True)

    scores_path = out / "scores.jsonl"
    done = set()
    if scores_path.exists():
        for line in open(scores_path):
            row = json.loads(line)
            done.add((row["key"], row["model"], row["repeat"]))
    write_lock = threading.Lock()

    def grade(answer: dict, model: str, repeat: int) -> None:
        task = tasks[answer["problem_idx"]]
        prompt = f"question : {task.get('core_question') or task['question']}\nmodel response : {answer['answer']}\nlabel : {task['ground_truth']}"
        started = time.time()
        score, rubric = finqa_eval._call_judge(finqa_eval.MULTI_TABLE_CORRECTNESS_PROMPT, prompt, multi_table=True)
        row = {"key": answer["key"], "model": model, "repeat": repeat, "score": float(score),
               "rubric": {k: rubric.get(k) for k in finqa_eval.CORRECTNESS_WEIGHTS}, "seconds": round(time.time() - started, 1)}
        with write_lock, open(scores_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")

    for model in args.models.split(","):
        # _call_judge reads the module-level model name at call time: one model per phase.
        finqa_eval.MULTI_TABLE_JUDGE_MODEL = model
        jobs = [(a, model, r) for a in answers for r in range(args.repeats) if (a["key"], model, r) not in done]
        print(f"{model}: {len(jobs)} gradings", flush=True)
        started = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            for i, _ in enumerate(pool.map(lambda job: grade(*job), jobs), 1):
                if i % 100 == 0:
                    print(f"  {model} {i}/{len(jobs)} {time.time() - started:.0f}s", flush=True)
    report(answers, scores_path, out)


def report(answers: list[dict], scores_path: Path, out: Path) -> None:
    by = defaultdict(list)
    rubric_by = defaultdict(lambda: defaultdict(list))
    for line in open(scores_path):
        row = json.loads(line)
        by[(row["model"], row["key"])].append(row["score"])
        for k, v in row["rubric"].items():
            if isinstance(v, int | float):
                rubric_by[(row["model"], row["key"])][k].append(v)
    train = {a["key"]: a["train_score"] for a in answers}
    lines = []
    for model in sorted({m for m, _ in by}):
        keys = [k for (m, k) in by if m == model and len(by[(m, k)]) >= 2]
        means = [st.mean(by[(model, k)]) for k in keys]
        stds = sorted(st.pstdev(by[(model, k)]) for k in keys)
        ranges = sorted(max(by[(model, k)]) - min(by[(model, k)]) for k in keys)
        within = st.mean(st.pvariance(by[(model, k)]) for k in keys)
        between = st.pvariance(means)
        n = len(keys)
        lines.append(f"== {model}: {n} answers x {len(by[(model, keys[0])])} gradings")
        lines.append(f"  score mean {st.mean(means):.3f}; between-answer std {between ** .5:.3f}; judge noise std (within answer) {within ** .5:.3f}"
                     f" -> noise share of total variance {within / (within + between):.0%}")
        lines.append(f"  per-answer std p50/p90/max {stds[n // 2]:.3f}/{stds[int(n * .9)]:.3f}/{stds[-1]:.3f}; range p50/p90/max {ranges[n // 2]:.2f}/{ranges[int(n * .9)]:.2f}/{ranges[-1]:.2f}")
        lines.append(f"  answers with range >= 0.2: {sum(r >= 0.2 for r in ranges)}/{n}; >= 0.3: {sum(r >= 0.3 for r in ranges)}/{n}")
        lines.append(f"  vs base avg@4 within-task variance (val 0.096^2 = 0.0092, test 0.097^2 = 0.0094): judge noise = {within / 0.0093:.0%} of it")
        first = [by[(model, k)][0] for k in keys]
        lines.append(f"  corr(first grading, training score) {correlation(first, [train[k] for k in keys]):.3f}; corr(mean of gradings, training score) {correlation(means, [train[k] for k in keys]):.3f}")
        per_bin = defaultdict(list)
        for key in keys:
            per_bin[min(9, int(st.mean(by[(model, key)]) * 10))].append(st.pstdev(by[(model, key)]))
        lines.append("  judge noise std by mean-score bin: " + "  ".join(
            f"{b / 10:.1f}-{(b + 1) / 10:.1f}: {st.mean(v):.3f} (n={len(v)})" for b, v in sorted(per_bin.items())))
        for k in finqa_eval.CORRECTNESS_WEIGHTS:
            comp = [st.pstdev(rubric_by[(model, key)][k]) for key in keys if len(rubric_by[(model, key)][k]) >= 2]
            if comp:
                lines.append(f"    component {k:<24} mean within-answer std {st.mean(comp):5.1f} (0-100 scale), weight {finqa_eval.CORRECTNESS_WEIGHTS[k]}")
    models = sorted({m for m, _ in by})
    if len(models) == 2:
        common = [k for (m, k) in by if m == models[0] and (models[1], k) in by]
        a = [st.mean(by[(models[0], k)]) for k in common]
        b = [st.mean(by[(models[1], k)]) for k in common]
        lines.append(f"== {models[0]} vs {models[1]} on {len(common)} answers: corr of mean scores {correlation(a, b):.3f}; mean diff {st.mean(x - y for x, y in zip(a, b)):+.3f}")
    text = "\n".join(lines)
    print(text)
    (out / "report.txt").write_text(text + "\n")


def correlation(x: list[float], y: list[float]) -> float:
    mx, my = st.mean(x), st.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


if __name__ == "__main__":
    main()
