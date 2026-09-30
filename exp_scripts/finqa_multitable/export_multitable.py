"""Write the FinQA multi-table train/val/test tasks as JSONL for the SAO driver and the evaluator.

The rows are finqa/data/multi_table_data/{train,val,test}_finqa.csv, built into tasks with
finqa/export_finqa.py's preprocess_fn (rllm prepare_finqa_data.py verbatim). These are exactly
rllm's multi_train / multi_val / multi_test (991 / 126 / 131): same question ids, and with
--verify every field the policy or judge reads is compared against rllm's registered parquet.

usage: export_multitable.py [--verify RLLM_DATASET_DIR]
  (RLLM_DATASET_DIR: e.g. rllm/exp_scripts/finqa-grpo-run/.rllm_multilane0_multi_val/datasets/finqa)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "finqa_singletable"))
from export_finqa import preprocess_fn, to_json_value  # noqa: E402

SOURCE = HERE.parent / "finqa_singletable" / "data" / "multi_table_data"
OUT = HERE / "data"
SPLITS = {"multi_train": ("train", 991), "multi_val": ("val", 126), "multi_test": ("test", 131)}
COMPARED = ("question", "ground_truth", "company", "question_id", "question_type", "core_question", "table_name")


def main() -> None:
    verify = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--verify" else None
    OUT.mkdir(exist_ok=True)
    for split, (csv_split, expected) in SPLITS.items():
        df = pd.read_csv(SOURCE / f"{csv_split}_finqa.csv")
        tasks = [{k: to_json_value(v) for k, v in preprocess_fn(row).items()} for _, row in df.iterrows()]
        if len(tasks) != expected:
            raise SystemExit(f"{split}: expected {expected} tasks (rllm's split), found {len(tasks)}")
        if any(not str(t["question_type"]).startswith("multi_table") for t in tasks):
            raise SystemExit(f"{split}: expected only multi_table question types")
        if verify is not None:
            reference = pd.read_parquet(verify / f"{split}.parquet")
            for idx, (task, (_, row)) in enumerate(zip(tasks, reference.iterrows())):
                for key in COMPARED:
                    ours, theirs = task[key], row[key]
                    theirs = [str(x) for x in theirs] if key == "table_name" else theirs
                    if (list(map(str, ours)) if key == "table_name" else ours) != theirs:
                        raise SystemExit(f"{split}[{idx}].{key}: expected rllm's {theirs!r}, got {ours!r}")
        out = OUT / f"{split}.jsonl"
        with open(out, "w") as fh:
            for idx, task in enumerate(tasks):
                fh.write(json.dumps({"problem_idx": idx, "split": split, "task": task}) + "\n")
        print(f"{split}: {len(tasks)} tasks -> {out}" + (" (fields match rllm's parquet)" if verify else ""))


if __name__ == "__main__":
    main()
