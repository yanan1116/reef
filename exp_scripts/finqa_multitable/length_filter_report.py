"""Length-filter report of a FinQA multi-table SAO run (length-filtered SAO, samples <= 16,384 tokens).

The driver does not train an episode whose assembled sample exceeds SAO_MAX_SAMPLE_TOKENS
(dropped "too_long", not re-queued). That is a sample-selection rule, so every comparison must
state how much was filtered and which tasks. This reports, from records.jsonl and the task pool:

  1. filter rate: too_long / (reported + too_long), overall and per training step (64 reported
     episodes per step), next to the version-straddle drops
  2. which tasks: question type, ground-truth answer length, number of tables, company, against
     the 991-task pool; tasks filtered every time they were sampled
  3. what the filtered episodes looked like: sample tokens, turns, endings
  5. version-straddle drops by sample length: the streaming driver drops (and re-queues) every
     episode that spans a weight publication; long episodes almost always do, so these drops are
     a second, larger length selection (2026-09-29 gate: 41% dropped, 22% at 4-6k tokens, 90% above
     12k; single-table: 2.9%, no length bias). Also tasks dropped and never trained.
  4. with --grade (on the training host: the Reef record store is host-local): the rubric score
     each filtered episode would have received (same judge), from its final answer rebuilt out
     of the recorded turns, against the reported episodes and the same tasks' reported episodes

usage: length_filter_report.py RESULTS_DIR [--grade STATE_DIR]
"""

from __future__ import annotations

import glob
import json
import shutil
import sqlite3
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = 64


def q(values: list[float], f: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(f * len(ordered)))]


def table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    line = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    return "\n".join([line(rows[0]), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in rows[1:]])


def rebuilt_answers(state: Path, episodes: list[dict]) -> dict[str, str]:
    """The flow's final answer of each episode: the last non-empty assistant content (its fallback rule)."""
    wanted = {rid for ep in episodes for rid in ep["agent_record_ids"]}
    with tempfile.TemporaryDirectory() as tmp:
        for f in glob.glob(str(state / "agent-record" / "*.sqlite3*")):
            shutil.copy(f, tmp)
        db = sqlite3.connect(glob.glob(tmp + "/*.sqlite3")[0])
        payloads = {rid: json.loads(p) for rid, p in db.execute("select agent_record_id, payload_json from agent_record where request_type = 'inference'") if rid in wanted}
        db.close()
    answers = {}
    for ep in episodes:
        contents = [(payloads[r]["response"]["choices"][0]["message"].get("content") or "") for r in ep["agent_record_ids"] if r in payloads]
        nonempty = [c for c in contents if c.strip()]
        answers[str(ep["position"])] = nonempty[-1] if nonempty else ""
    return answers


def main() -> None:
    out = Path(sys.argv[1])
    grade_state = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--grade" else None
    records = [json.loads(line) for line in open(out / "records.jsonl") if line.strip()]
    pool = {json.loads(l)["problem_idx"]: json.loads(l)["task"] for l in open(HERE / "data" / "multi_train.jsonl")}
    reported = [r for r in records if not r.get("dropped")]
    too_long = [r for r in records if r.get("dropped") == "too_long"]
    straddle = [r for r in records if r.get("dropped") and r["dropped"] != "too_long"]
    finished = len(reported) + len(too_long)

    print(f"Length-filter report: {out.name} (length-filtered SAO, samples <= 16,384 tokens)")
    print(f"1. Filter rate: too_long {len(too_long)}/{finished} = {100 * len(too_long) / max(finished, 1):.2f}% of finished assemblable "
          f"episodes; version-straddle drops {len(straddle)} (re-queued, not a selection rule)")
    ordered = sorted(records, key=lambda r: r["recorded_at"])
    per_step: dict[int, Counter] = defaultdict(Counter)
    count = 0
    for r in ordered:
        step = count // BATCH + 1
        if not r.get("dropped"):
            count += 1
            per_step[step]["reported"] += 1
        elif r["dropped"] == "too_long":
            per_step[step]["too_long"] += 1
    print()
    print("5. Version-straddle drops by assembled sample length (re-queued, but long episodes rarely get trained)")
    straddle_rows = [["Sample tokens", "Finished", "Straddle-dropped", "Drop rate"]]
    for lo, hi in [(0, 4000), (4000, 6000), (6000, 8000), (8000, 12000), (12000, 16384), (16384, 10**9)]:
        bucket = [r for r in reported + straddle if lo <= (r.get("sample_tokens") or 0) < hi]
        dropped_here = [r for r in bucket if r.get("dropped")]
        straddle_rows.append([f"{lo}-{hi if hi < 10**9 else 'max'}", str(len(bucket)), str(len(dropped_here)),
                              f"{100 * len(dropped_here) / max(len(bucket), 1):.0f}%"])
    print(table(straddle_rows))
    dropped_tasks = Counter(r["problem_idx"] for r in straddle)
    trained_tasks = Counter(r["problem_idx"] for r in reported)
    print(f"   tasks straddle-dropped at least once: {len(dropped_tasks)}; never trained yet: "
          f"{sum(trained_tasks[k] == 0 for k in dropped_tasks)}; dropped >= 2 times and never trained: "
          f"{sum(v >= 2 and trained_tasks[k] == 0 for k, v in dropped_tasks.items())}; "
          f"longest trained sample {max((r.get('sample_tokens') or 0) for r in reported)} tokens")
    print()
    rows = [["Steps", "Reported", "too_long", "Filter rate"]]
    steps = sorted(per_step)
    for start in range(0, len(steps), 10):
        chunk = steps[start:start + 10]
        rep = sum(per_step[s]["reported"] for s in chunk)
        tl = sum(per_step[s]["too_long"] for s in chunk)
        rows.append([f"{chunk[0]}-{chunk[-1]}", str(rep), str(tl), f"{100 * tl / max(rep + tl, 1):.2f}% ({tl}/{rep + tl})"])
    print(table(rows))
    if not too_long:
        return

    def describe(tasks: list[dict]) -> list[str]:
        types = Counter(t["question_type"] for t in tasks)
        gt = [len(t["ground_truth"]) for t in tasks]
        tables = [len(t["table_name"]) for t in tasks]
        return [f"{100 * types.get('multi_table_hard', 0) / len(tasks):.1f}%", f"{statistics.median(gt):.0f}",
                f"{q(gt, .9):.0f}", f"{statistics.mean(tables):.2f}"]

    print()
    print("2. Which tasks (filtered episodes vs the 991-task pool vs reported episodes)")
    rows = [["Set", "N", "Hard share", "GT chars median", "GT chars p90", "Tables mean"]]
    for name, tasks in [("pool (991 tasks)", list(pool.values())), ("reported episodes", [pool[r["problem_idx"]] for r in reported]),
                        ("too_long episodes", [pool[r["problem_idx"]] for r in too_long])]:
        rows.append([name, str(len(tasks))] + describe(tasks))
    print(table(rows))
    seen = Counter(r["problem_idx"] for r in reported + too_long)
    filtered = Counter(r["problem_idx"] for r in too_long)
    always = [k for k, v in filtered.items() if v == seen[k]]
    print(f"   distinct tasks filtered: {len(filtered)}; filtered every time sampled: {len(always)} "
          f"(of which sampled >= 2 times: {sum(seen[k] >= 2 for k in always)}); companies: {dict(Counter(pool[k]['company'] for k in filtered).most_common(8))}")

    print()
    print("3. What the filtered episodes looked like")
    rows = [["Set", "Sample tokens median", "p90", "max", "Turns median", "Endings"]]
    for name, eps in [("reported", reported), ("too_long", too_long)]:
        toks = [r.get("sample_tokens") or 0 for r in eps]
        rows.append([name, f"{statistics.median(toks):.0f}", f"{q(toks, .9):.0f}", str(max(toks)),
                     f"{statistics.median(r['turns'] for r in eps):.0f}", json.dumps(dict(Counter(r["ended"] for r in eps)))])
    print(table(rows))

    if grade_state is None:
        return
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(HERE.parent / "finqa_singletable"))
    import os
    os.environ.setdefault("FINQA_MULTI_TABLE_JUDGE_MODEL", "gpt-5.4-nano")
    from judge_env import load_judge_env
    load_judge_env()
    import finqa_env
    answers = rebuilt_answers(grade_state, too_long)
    scores = []
    for r in too_long:
        stand_in = sys.modules["rllm.types"].Episode(artifacts={"answer": answers[str(r["position"])], "accessed_tables": [], "turns": r["turns"]})
        scores.append(float(finqa_env.finqa_eval.finqa_evaluator(pool[r["problem_idx"]], stand_in).reward))
    same = [x["score"] for x in reported if x["problem_idx"] in filtered]
    print()
    print("4. Scores the filtered episodes would have received (post hoc, same judge; table access not rebuilt)")
    rows = [["Set", "N", "Mean score", "Score >= 0.9"],
            ["too_long (graded post hoc)", str(len(scores)), f"{statistics.mean(scores):.3f}", f"{sum(s >= 0.9 for s in scores)}/{len(scores)}"],
            ["reported, same tasks", str(len(same)), f"{statistics.mean(same):.3f}" if same else "-", f"{sum(s >= 0.9 for s in same)}/{len(same)}" if same else "-"],
            ["reported, all", str(len(reported)), f"{statistics.mean(r['score'] for r in reported):.3f}", f"{sum(r['score'] >= 0.9 for r in reported)}/{len(reported)}"]]
    print(table(rows))


if __name__ == "__main__":
    main()
