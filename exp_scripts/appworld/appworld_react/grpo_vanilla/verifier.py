"""Canonical Claw-Eval-Live scoring for one finished rollout.

The benchmark owns grading.  This module loads the task's own grader from the
task directory, calls it with exactly the arguments the benchmark CLI passes,
and aggregates with the benchmark's own ``compute_task_score`` / ``is_pass``.
No reward shaping happens anywhere in this file.

The distinction the baseline depends on:

* a grader that runs and returns a score of ``0.0`` is a *real* score and a
  valid learning signal;
* a grader that cannot run, or a judge that cannot answer, is an *infrastructure
  failure* and yields ``None``, which rejects the whole group.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import sha256_text
from .judge import JudgeFailure

PASS_THRESHOLD = 0.75


class VerifierFailure(RuntimeError):
    """The canonical grader could not produce a score."""


@dataclass
class VerifierOutcome:
    """Result of grading one rollout."""

    score: float | None
    passed: bool | None
    status: str
    components: dict[str, Any]
    judge_calls: list[dict[str, Any]]
    error: str | None = None

    def record(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "status": self.status,
            "components": self.components,
            "judge_calls": self.judge_calls,
            "error": self.error,
        }


def load_grader(task_id: str, task_dir: Path, tasks_dir: Path) -> Any:
    """The task's own grader class, loaded from the task directory."""
    from .benchmark import active

    get_grader = active().module("graders.registry").get_grader

    return get_grader(task_id, tasks_dir=tasks_dir, task_dir=task_dir)


def grade_rollout(
    *,
    task: Any,
    task_id: str,
    task_dir: Path,
    tasks_dir: Path,
    messages: list[Any],
    dispatches: list[Any],
    audit_data: dict[str, Any] | None,
    env_snapshot: dict[str, Any] | None,
    judge: Any | None,
    judge_mode: str,
) -> VerifierOutcome:
    """Run the canonical grader for one trajectory.

    ``judge=None`` is only legitimate in an explicitly disabled-judge dev run;
    the resulting components are tagged so those numbers can never be mistaken
    for judged scores.
    """
    from .benchmark import active

    scoring = active().module("models.scoring")
    compute_task_score, is_pass = scoring.compute_task_score, scoring.is_pass

    judge_records: list[dict[str, Any]] = []
    try:
        grader = load_grader(task_id, task_dir, tasks_dir)
    except Exception as exc:
        raise VerifierFailure(f"{task_id}: grader could not be loaded: {exc!r}") from exc

    parameters = inspect.signature(grader.grade).parameters
    kwargs: dict[str, Any] = {"audit_data": audit_data, "judge": judge}
    if "media_events" in parameters:
        kwargs["media_events"] = []
    if "env_snapshot" in parameters and env_snapshot is not None:
        kwargs["env_snapshot"] = env_snapshot

    try:
        scores = grader.grade(messages, dispatches, task, **kwargs)
    except JudgeFailure as exc:
        if judge is not None:
            judge_records = judge.records()
        return VerifierOutcome(
            score=None,
            passed=None,
            status="judge_failure",
            components={"judge_mode": judge_mode},
            judge_calls=judge_records,
            error=repr(exc),
        )
    except Exception as exc:  # noqa: BLE001 - a grader crash is an infra failure
        if judge is not None:
            judge_records = judge.records()
        return VerifierOutcome(
            score=None,
            passed=None,
            status="grader_failure",
            components={"judge_mode": judge_mode},
            judge_calls=judge_records,
            error=repr(exc),
        )

    if judge is not None:
        judge_records = judge.records()
    task_score = float(compute_task_score(scores))
    components = {
        "judge_mode": judge_mode,
        "grader_class": type(grader).__name__,
        "grader_module_sha256": _grader_source_hash(task_dir),
        "dimensions": {
            "completion": float(scores.completion),
            "robustness": float(scores.robustness),
            "communication": float(scores.communication),
            "safety": float(scores.safety),
            "efficiency_turns": int(scores.efficiency_turns),
            "efficiency_tokens": int(scores.efficiency_tokens),
        },
        "aggregation": f"{active().package}.models.scoring.compute_task_score",
        "pass_threshold": PASS_THRESHOLD,
        "judge_call_count": len(judge_records),
    }
    return VerifierOutcome(
        score=task_score,
        passed=bool(is_pass(task_score, PASS_THRESHOLD)),
        status="scored",
        components=components,
        judge_calls=judge_records,
    )


def _grader_source_hash(task_dir: Path) -> str | None:
    grader = Path(task_dir) / "grader.py"
    if not grader.is_file():
        return None
    return sha256_text(grader.read_text(encoding="utf-8"))
