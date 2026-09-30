"""Audit report of a FinQA multi-table SAO smoke run: what was actually sent, trained and measured.

Run on the training host after the smoke (the Reef record store and Slime log are host-local).
Complements smoke_check.py (pass/fail gates) with evidence a reviewer can read:

  A. Protocol conformance, every inference record of every reported episode, read from Reef's
     own record store (a snapshot copy of the SQLite files, which are root-owned):
       sampling (temperature 0.7, top_p 1.0), tool list == TOOL_SPECS on every call,
       max_tokens 2048 in the tool phase / 8192 after "Tool use is now closed",
       prompt + max_tokens <= 49152, first two messages == the flow's builder output,
       mid-conversation system messages only the flow's literals, each turn's prompt extends
       the previous turn's prompt + reply, one weight version per turn,
       tokens = prompt + response and loss mask = response length (trained span).
     Reports: the reported score equals records.jsonl's, references = the episode's turns in order.
  B. Episodes: score distribution, turns, tool use, endings, drops, prompt-count agreement, time.
  C. Training, per step from the Slime driver log: rollout reward and lengths, actor loss,
     clip fraction, KL, train-vs-rollout log-prob gap (bf16 logits vs the fp32 single-table run),
     grad norm, critic loss / explained variance, step time; GPU memory from the monitor.
  D. One episode transcript, abridged, for a human read.

usage: smoke_audit.py RESULTS_DIR STATE_DIR [MONITOR_TSV]
"""

from __future__ import annotations

import glob
import os
import json
import math
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "finqa_singletable"))
from judge_env import load_judge_env  # noqa: E402

load_judge_env()
import multitable_env as env  # noqa: E402

TOOL_SPECS_JSON = json.dumps(env.TOOL_SPECS, sort_keys=True)
FLOW_SYSTEM_LITERALS = {env.MALFORMED_TOOL_CALL_MESSAGE, env.TOOL_USE_CLOSED_MESSAGE, env.EMPTY_FINAL_MESSAGE}
CONTEXT = 49152
SINGLE_TABLE_LOG = "/home/yanan/reef-sao-finqa/state/stack/slime-driver.log"  # fp32-logits reference


def load_records(state: Path) -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        for f in glob.glob(str(state / "agent-record" / "*.sqlite3*")):
            shutil.copy(f, tmp)
        db = sqlite3.connect(glob.glob(tmp + "/*.sqlite3")[0])
        rows = db.execute("select agent_record_id, request_type, payload_json, references_json from agent_record").fetchall()
        db.close()
    return {rid: {"type": t, "payload": json.loads(p), "references": json.loads(r)} for rid, t, p, r in rows}


def is_budget_checkpoint(text: str) -> bool:
    return (text.startswith(env.BUDGET_CHECKPOINT_PREFIX) and text.endswith(env.BUDGET_CHECKPOINT_SUFFIX)
            and text[len(env.BUDGET_CHECKPOINT_PREFIX):-len(env.BUDGET_CHECKPOINT_SUFFIX)].isdigit())


def audit_protocol(episodes: list[dict], tasks: dict[int, dict], records: dict[str, dict]) -> tuple[Counter, int]:
    problems: Counter = Counter()
    calls = 0
    for ep in episodes:
        task = tasks[ep["problem_idx"]]
        expected_head = env.flow.build_policy_visible_initial_messages(task, task["question"], str(task["question_id"]))
        previous: list[dict] | None = None
        for rid in ep["agent_record_ids"]:
            rec = records.get(rid)
            if rec is None:
                problems["record missing from the store"] += 1
                continue
            p, calls = rec["payload"], calls + 1
            msgs, resp = p["messages"], p["response"]
            train = resp.get("training") or {}
            if p.get("temperature") != 0.7 or p.get("top_p") != 1.0:
                problems["sampling != (0.7, 1.0)"] += 1
            if json.dumps(p.get("tools"), sort_keys=True) != TOOL_SPECS_JSON:
                problems["tool list != TOOL_SPECS"] += 1
            closed = any(m.get("role") == "system" and m.get("content") == env.TOOL_USE_CLOSED_MESSAGE for m in msgs)
            if p.get("max_tokens") != (env.flow.FINAL_MAX_COMPLETION_TOKENS if closed else env.flow.DISCOVERY_MAX_COMPLETION_TOKENS):
                problems["max_tokens not the flow's cap for its phase"] += 1
            if train.get("prompt_length", 0) + p.get("max_tokens", 0) > CONTEXT:
                problems["prompt + max_tokens > 49152 was sent"] += 1
            if msgs[:2] != expected_head:
                problems["first two messages != the flow's builder output"] += 1
            for m in msgs[2:]:
                if m.get("role") == "system" and m.get("content") not in FLOW_SYSTEM_LITERALS and not is_budget_checkpoint(m.get("content", "")):
                    problems["unknown mid-conversation system message"] += 1
            if previous is not None and msgs[:len(previous)] != previous:
                problems["turn prompt does not extend the previous turn"] += 1
            reply = env.history_message(resp["choices"][0]["message"])
            previous = msgs + [reply]
            spans = train.get("runtime_load_spans") or []
            if len({s["runtime_load_id"] for s in spans}) != 1:
                problems["turn not from exactly one weight version"] += 1
            if len(train.get("tokens", [])) != train.get("prompt_length", -1) + train.get("response_length", -1):
                problems["tokens != prompt + response"] += 1
            if sum(train.get("loss_mask", [])) != train.get("response_length"):
                problems["loss mask != response length"] += 1
            if not all(math.isfinite(x) for x in train.get("rollout_log_probs", [])):
                problems["non-finite rollout log-prob"] += 1
        report = [r for r in records.values() if r["type"] == "report" and r["references"] == ep["agent_record_ids"]]
        if len(report) != 1:
            problems["episode without exactly one report referencing its turns in order"] += 1
        elif abs(float(report[0]["payload"].get("score", report[0]["payload"].get("feedback", {}).get("score", -1))) - ep["score"]) > 1e-9:
            problems["reported score != records.jsonl score"] += 1
    return problems, calls


def slime_metrics(log_path: str) -> dict[str, dict[int, dict]]:
    series: dict[str, dict[int, dict]] = {"rollout": {}, "step": {}, "critic-step": {}, "perf": {}}
    pattern = re.compile(r" - (rollout|step|critic-step|perf) (\d+): (\{.*\})\s*$")
    with open(log_path, errors="replace") as fh:
        for line in fh:
            found = pattern.search(line)
            if found:
                try:
                    series[found.group(1)][int(found.group(2))] = eval(found.group(3), {"__builtins__": {}}, {"nan": float("nan"), "inf": float("inf")})
                except SyntaxError:
                    pass
    return series


def main() -> None:
    out, state = Path(sys.argv[1]), Path(sys.argv[2])
    monitor = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    lines: list[str] = []
    say = lines.append
    rows = [json.loads(line) for line in open(out / "records.jsonl") if line.strip()]
    reported = [r for r in rows if not r.get("dropped")]
    dropped = [r for r in rows if r.get("dropped")]
    tasks = {json.loads(l)["problem_idx"]: json.loads(l)["task"] for l in open(HERE / "data" / "multi_train.jsonl")}
    records = load_records(state)

    say(f"FinQA multi-table SAO smoke audit: {out.name}")
    say("=" * 100)
    problems, calls = audit_protocol(reported, tasks, records)
    say(f"A. Protocol conformance: {len(reported)} reported episodes, {calls} model calls read from Reef's record store")
    checks = ["sampling != (0.7, 1.0)", "tool list != TOOL_SPECS", "max_tokens not the flow's cap for its phase",
              "prompt + max_tokens > 49152 was sent", "first two messages != the flow's builder output",
              "unknown mid-conversation system message", "turn prompt does not extend the previous turn",
              "turn not from exactly one weight version", "tokens != prompt + response", "loss mask != response length",
              "non-finite rollout log-prob", "record missing from the store",
              "episode without exactly one report referencing its turns in order", "reported score != records.jsonl score"]
    for name in checks + [k for k in problems if k not in checks]:
        say(f"   [{'OK  ' if problems[name] == 0 else 'FAIL'}] {name}: {problems[name]}")

    scores = sorted(r["score"] for r in reported)
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]
    say("")
    say("B. Episodes")
    say(f"   rubric score: mean {statistics.mean(scores):.3f}, median {q(scores, .5):.3f}, p10 {q(scores, .1):.3f}, "
        f"p90 {q(scores, .9):.3f}, max {scores[-1]:.3f}; score >= 0.9: {sum(s >= 0.9 for s in scores)}/{len(scores)}; "
        f"score == 0: {sum(s == 0 for s in scores)}/{len(scores)}")
    turns = sorted(r["turns"] for r in reported)
    say(f"   turns: median {q(turns, .5)}, p90 {q(turns, .9)}, max {turns[-1]}; tool calls mean "
        f"{statistics.mean(r['tool_calls'] for r in reported):.1f}; tool errors {sum(r['tool_errors'] for r in reported)}/"
        f"{sum(r['tool_calls'] for r in reported)}; malformed tool calls {sum(r['malformed_tool_calls'] for r in reported)}")
    say(f"   endings: {dict(Counter(r['ended'] for r in reported))}; fallback answers {sum(r['final_fallback_used'] for r in reported)}; "
        f"episodes with a refused call {sum(bool(r['llm_errors']) for r in reported)}")
    prompts = sorted(r["prompt_tokens_last"] or 0 for r in reported)
    say(f"   last-turn prompt tokens: median {q(prompts, .5)}, p90 {q(prompts, .9)}, max {prompts[-1]}; completion tokens "
        f"mean {statistics.mean(r['completion_tokens'] for r in reported):.0f}")
    say(f"   dropped before reporting: {len(dropped)}/{len(rows)} {dict(Counter(r['dropped'] for r in dropped))}; "
        f"local vs engine prompt-count mismatches: {sum(r['prompt_count_mismatches'] for r in rows)}/{sum(r['turns'] for r in rows)} turns")
    secs = sorted(r["seconds"] for r in reported)
    say(f"   seconds per episode: median {q(secs, .5):.0f}, p90 {q(secs, .9):.0f}, max {secs[-1]:.0f}")

    say("")
    say("C. Training (Slime driver log)")
    m = slime_metrics(str(state / "stack" / "slime-driver.log"))
    say(f"   {'step':>4} {'reward':>7} {'resp_len':>8} {'tot_len':>8} {'trunc':>5} {'loss':>8} {'clipfrac':>8} {'ppo_kl':>8} "
        f"{'lp_gap':>7} {'grad':>6} {'critic_vl':>9} {'expl_var':>8} {'step_s':>7}")
    for step in sorted(set(m["rollout"]) | set(m["critic-step"]) | set(m["step"])):
        r, a, c, p = m["rollout"].get(step, {}), m["step"].get(step, {}), m["critic-step"].get(step, {}), m["perf"].get(step, {})
        f = lambda d, k, w, fmt: format(d[k], fmt).rjust(w) if k in d else "-".rjust(w)
        say(f"   {step:>4} {f(r, 'rollout/rewards', 7, '.3f')} {f(r, 'rollout/response_lengths', 8, '.0f')} "
            f"{f(r, 'rollout/total_lengths', 8, '.0f')} {f(r, 'rollout/truncated', 5, '.2f')} {f(a, 'train/loss', 8, '.4f')} "
            f"{f(a, 'train/pg_clipfrac', 8, '.4f')} {f(a, 'train/ppo_kl', 8, '.4f')} {f(a, 'train/train_rollout_logprob_abs_diff', 7, '.4f')} "
            f"{f(a, 'train/grad_norm', 6, '.3f')} {f(c, 'train/critic-value_loss', 9, '.4f')} "
            f"{f(c, 'train/critic-explained_variance', 8, '.3f')} {f(p, 'perf/step_time', 7, '.0f')}")
    gaps = [v["train/train_rollout_logprob_abs_diff"] for v in m["step"].values() if "train/train_rollout_logprob_abs_diff" in v]
    if Path(SINGLE_TABLE_LOG).exists():
        ref = slime_metrics(SINGLE_TABLE_LOG)["step"]
        ref_gaps = [ref[s]["train/train_rollout_logprob_abs_diff"] for s in sorted(ref)[:10] if "train/train_rollout_logprob_abs_diff" in ref[s]]
        if gaps and ref_gaps:
            say(f"   train-vs-rollout |log-prob| gap: this run (REEF_BF16_LOGITS={os.environ.get('REEF_BF16_LOGITS', '0')}) mean {statistics.mean(gaps):.4f}; "
                f"single-table first {len(ref_gaps)} actor steps (fp32 logits) mean {statistics.mean(ref_gaps):.4f}")
    if monitor and monitor.exists():
        mem = [line.split("\t") for line in monitor.read_text().splitlines()[1:] if out.name in line]
        if mem:
            say(f"   GPU memory peak over {len(mem)} monitor samples: gpu0 {max(int(x[3]) for x in mem)} MiB, "
                f"gpu1 {max(int(x[4]) for x in mem)} MiB (of 49140)")

    say("")
    say("D. One episode, abridged (the highest-scoring reported one)")
    best = max(reported, key=lambda r: r["score"])
    last = records[best["agent_record_ids"][-1]]["payload"]
    say(f"   problem_idx {best['problem_idx']} score {best['score']:.3f} turns {best['turns']} ended {best['ended']}")
    for msg in last["messages"] + [env.history_message(last["response"]["choices"][0]["message"])]:
        text = msg.get("content") or ""
        calls_text = "; ".join(f"{c['function']['name']}({c['function']['arguments'][:120]})" for c in msg.get("tool_calls") or [])
        say(f"   [{msg['role']}] {(text[:300] + ' ...') if len(text) > 300 else text}{('  CALLS ' + calls_text) if calls_text else ''}".replace("\n", " | "))
    report = "\n".join(lines)
    (out / "audit_report.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
