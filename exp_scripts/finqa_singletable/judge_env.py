"""Load the FinQA judge settings of the PRPO runs before finqa_env is imported.

Mirrors gitlab/tail/rllm/finqa-grpo-run/env.sh: the credentials file (mode 600,
never versioned; default exp_scripts/finqa/.judge_creds) holds
`export OPENAI_API_KEY=...` and `export OPENAI_BASE_URL=...`; the judge is
gpt-5.4-nano (multi-table: gpt-5.4-mini) with up to 10 retries on a non-"stop"
finish. Existing environment values win, so a launcher can override any of them.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CREDS = Path(__file__).resolve().parent / ".judge_creds"


def load_judge_env(creds_path: str | os.PathLike | None = None) -> None:
    path = Path(creds_path or os.environ.get("FINQA_JUDGE_CREDS") or DEFAULT_CREDS)
    if not path.is_file():
        raise FileNotFoundError(f"judge credentials not found: {path} (set FINQA_JUDGE_CREDS)")
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            key, _, value = line[len("export "):].partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        if not os.environ.get(key):
            raise RuntimeError(f"{key} is missing from {path}")
    os.environ.setdefault("FINQA_JUDGE_MODEL", "gpt-5.4-nano")
    os.environ.setdefault("FINQA_MULTI_TABLE_JUDGE_MODEL", "gpt-5.4-mini")
    os.environ.setdefault("FINQA_JUDGE_MAX_ATTEMPTS", "10")
