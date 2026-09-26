"""Write the FinQA train/val/test tasks as JSONL for the SAO driver and the evaluator.

Each task is built exactly as rllm's cookbooks/finqa/prepare_finqa_data.py builds
the dataset PRPO trained and was evaluated on: preprocess_fn below is that
function verbatim (its _parse_json_list is parse_json_list here, same body), and
the CSVs are read with pandas the same way, so field values and their types
(which the judge prompt interpolates) match. One line per task: {"problem_idx", "split", "task"}; problem_idx is the
row position in its split.

usage: export_finqa.py [DATA_DIR]   (default exp_scripts/finqa/data)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPECTED = {"train": 4030, "val": 522, "test": 558}  # the PRPO splits


def parse_json_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = s
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            cleaned = parsed.strip()
            return [cleaned] if cleaned else []
        return []
    return []


def preprocess_fn(example: dict) -> dict:
    return {
        "question": example["user_query"],
        "ground_truth": example["answer"],
        "data_source": "finqa",
        "company": example["company"],
        "question_id": str(example["id"]),
        "question_type": example["question_type"],
        "core_question": example["question"],
        "table_name": parse_json_list(example.get("table_name")),
        "columns_used": parse_json_list(example.get("columns_used_json")),
        "rows_used": parse_json_list(example.get("rows_used_json")),
        "explanation": example["explanation"],
    }


def to_json_value(value):
    """numpy scalars -> Python scalars. NaN stays NaN (json round-trips it), as in the
    dataset PRPO read: the judge prompt prints it as "nan" and the evaluator treats it
    as present, so turning it into None would change both."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, list):
        return [to_json_value(v) for v in value]
    return value


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
    for split, expected in EXPECTED.items():
        df = pd.read_csv(data_dir / f"{split}_finqa.csv")
        tasks = [preprocess_fn(row) for _, row in df.iterrows()]
        if len(tasks) != expected:
            raise SystemExit(f"{split}: {len(tasks)} tasks, expected {expected} (the PRPO split)")
        out = data_dir / f"finqa_{split}.jsonl"
        with open(out, "w") as fh:
            for idx, task in enumerate(tasks):
                fh.write(json.dumps({"problem_idx": idx, "split": split, "task": {k: to_json_value(v) for k, v in task.items()}}) + "\n")
        print(f"{split}: {len(tasks)} tasks -> {out}")


if __name__ == "__main__":
    main()
