"""Write the DeepCoder training pool in stream.py's problems-JSONL shape.

Mirrors recipes/sao/examples/imo_answerbench/export_problems.py: one line per
problem with ``problem_idx`` and ``problem``. There is no ``gold``: DeepCoder is
graded by running hidden tests, which deepcoder_grader.py looks up by index, so
the tests never leave the rLLM dataset. Only the train split is exported; the
687-task test split stays held out for evaluation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepcoder_grader import _rows  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "deepcoder_train.jsonl")
rows = _rows()
with open(out, "w") as handle:
    for idx, row in enumerate(rows):
        handle.write(json.dumps({"problem_idx": idx, "problem": str(row["question"])}) + "\n")
print(f"wrote {len(rows)} problems to {out}")
