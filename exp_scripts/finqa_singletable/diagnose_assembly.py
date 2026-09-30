"""Find the FinQA episodes Reef cannot assemble into one multi-turn sample, and say why.

Runs inside the reef image (Reef's own assembly code) against a copy of a run's
agent-record SQLite database. For every reported episode in records.jsonl it
calls make_multi_turn_policy_trajectory exactly as the SAO processor does; for
each failure it replays that function's checks and prints the first one that
fails, and for a token fork the decoded text on both sides of the divergence.

usage (in the container): diagnose_assembly.py RECORD_DB RECORDS_JSONL [MODEL_DIR]
"""

from __future__ import annotations

import collections
import json
import sys

from transformers import AutoTokenizer

from reef.storage.sqlite import SQLiteRecordStore
from reef.train.processors.common import (
    _common_prefix_length,
    _drift_is_realignable,
    make_multi_turn_policy_trajectory,
    make_policy_trajectory,
)

SCENARIO = "sao-finqa"
REALIGN_THRESHOLD = 1024  # = sao_multiturn.MultiTurnSAORecipe
SCAFFOLD_TOLERANCE = 0


def explain(turns: list, tokenizer) -> str:
    """The first check of make_multi_turn_policy_trajectory that this episode fails."""
    versions = {turn.training.get("runtime_load_id") for turn in turns}
    if len(versions) != 1 or None in versions or "" in versions:
        return f"runtime_load_id across turns: {sorted(map(str, versions))}"
    has = [bool(turn.training.get("rollout_log_probs", [])) for turn in turns]
    if any(has) and not all(has):
        return f"rollout_log_probs present on turns {has}"
    tokens: list[int] = []
    latest_response_start = None
    for index, turn in enumerate(turns):
        mask = turn.training.get("loss_mask", [])
        full = turn.training.get("tokens", [])
        if not mask:
            return f"turn {index}: empty response (loss_mask length 0)"
        if len(full) <= len(mask):
            return f"turn {index}: tokens {len(full)} <= response {len(mask)}"
        prompt, output = list(full[: -len(mask)]), list(full[-len(mask):])
        if index > 0:
            common = _common_prefix_length(tokens, prompt)
            if common != len(tokens) and not _drift_is_realignable(
                common, latest_response_start, len(tokens), REALIGN_THRESHOLD, SCAFFOLD_TOLERANCE
            ):
                before = tokenizer.decode(tokens[max(0, common - 40) : common])
                ours = tokenizer.decode(tokens[common : common + 60])
                theirs = tokenizer.decode(prompt[common : common + 60])
                return (
                    f"turn {index}: fork at token {common}; latest response starts at {latest_response_start}, "
                    f"assembled length {len(tokens)}\n      context : {before!r}\n"
                    f"      assembled: {ours!r}\n      re-render: {theirs!r}"
                )
            if common != len(tokens):
                tokens = tokens[: min(latest_response_start, common)] + prompt[min(latest_response_start, common) :]
            else:
                tokens.extend(prompt[len(tokens) :])
        else:
            tokens.extend(prompt)
        latest_response_start = len(tokens)
        tokens.extend(output)
    return "no check failed in the replay (loss mask sums to 0?)"


def main() -> None:
    database, records_path = sys.argv[1], sys.argv[2]
    model_dir = sys.argv[3] if len(sys.argv) > 3 else "/root/models/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    store = SQLiteRecordStore(database)
    episodes = [json.loads(line) for line in open(records_path) if line.strip()]
    episodes = [e for e in episodes if "dropped" not in e]
    failures: collections.Counter = collections.Counter()
    missing = 0
    for episode in episodes:
        items = [store.get(SCENARIO, rid) for rid in episode["agent_record_ids"]]
        if any(item is None for item in items):
            missing += 1
            continue
        sample = make_multi_turn_policy_trajectory(
            items, float(episode["score"]), source_agent_record_id="diagnose",
            realign_threshold=REALIGN_THRESHOLD, scaffold_tolerance=SCAFFOLD_TOLERANCE,
        )
        if sample is not None:
            continue
        reason = explain([make_policy_trajectory(item, 0.0) for item in items], tokenizer)
        failures[reason.split(":")[0].split(" fork")[0]] += 1
        print(f"[FAIL] position={episode['position']} idx={episode['problem_idx']} turns={episode['turns']} "
              f"finish={episode['finish_reasons']}\n   {reason}")
    print(f"episodes {len(episodes)}, records missing {missing}, unassemblable {sum(failures.values())}: {dict(failures)}")


if __name__ == "__main__":
    main()
