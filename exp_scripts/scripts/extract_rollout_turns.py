"""Pull sampled turns (tokens + the engine's rollout log-probs) of given weight versions out of a
Reef record store, for adapter_fidelity.py. Run on the training host (the store is host-local
and root-owned: a snapshot copy of the SQLite files is read).

A turn is taken only if all its tokens came from one version (a single runtime_load_span).

usage: extract_rollout_turns.py AGENT_RECORD_DIR OUT_JSON VERSION[,VERSION...] [PER_VERSION] [MAX_TOKENS]
"""

from __future__ import annotations

import glob
import json
import shutil
import sqlite3
import sys
import tempfile


def main() -> None:
    record_dir, out_path = sys.argv[1], sys.argv[2]
    versions = [int(v) for v in sys.argv[3].split(",")]
    per_version = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
    picked: dict[int, list[dict]] = {v: [] for v in versions}
    with tempfile.TemporaryDirectory() as tmp:
        for f in glob.glob(record_dir + "/*.sqlite3*"):
            shutil.copy(f, tmp)
        db = sqlite3.connect(glob.glob(tmp + "/*.sqlite3")[0])
        query = ("select agent_record_id, payload_json from agent_record where request_type = 'inference' "
                 "and json_extract(payload_json, '$.runtime_load_id') like ? order by sequence")
        for version in versions:
            for rid, payload_json in db.execute(query, (f"%:{version}",)):
                train = json.loads(payload_json)["response"].get("training") or {}
                spans = train.get("runtime_load_spans") or []
                if len({s["runtime_load_id"] for s in spans}) != 1 or not str(spans[0]["runtime_load_id"]).endswith(f":{version}"):
                    continue
                if len(train["tokens"]) > max_tokens or train["response_length"] < 32:
                    continue
                picked[version].append({"agent_record_id": rid, "version": version, "tokens": train["tokens"],
                                        "prompt_length": train["prompt_length"],
                                        "rollout_log_probs": train["rollout_log_probs"]})
                if len(picked[version]) >= per_version:
                    break
        db.close()
    json.dump(picked, open(out_path, "w"))
    print({v: len(t) for v, t in picked.items()})


if __name__ == "__main__":
    main()
