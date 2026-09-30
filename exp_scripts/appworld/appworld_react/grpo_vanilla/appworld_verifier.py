"""Turn AppWorld's own evaluation into the ``VerifierOutcome`` the reward adapter expects.

AppWorld grades deterministically: it diffs the task's start and end databases and runs the
task's own unit tests from ``ground_truth/evaluation.py``. There is no rubric and no LLM
judge anywhere in that path, so this module never calls one -- ``judge``/``judge_mode`` are
accepted only because ``reward.ClawVerifierReward`` passes them for every benchmark.

The reward is the fraction of the task's requirement tests that passed, counting only the
tests a do-nothing agent would fail. AppWorld labels each test by how an empty agent --
one that calls nothing and changes nothing -- fares on it:

* ``no_op_fail``: the empty agent fails it, so passing it means the policy did the work;
* ``no_op_pass``: the empty agent passes it anyway. These are almost all side-effect
  guards ("assert no model changes"), which hold trivially when nothing was done.

Counting the second kind hands out free credit for accomplishing nothing. Measured on the
dev split, 18% of all tests carry ``no_op_pass``, and a Qwen3-4B run scored a mean of
0.248 there while passing *zero* ``no_op_fail`` tests -- every point came from not
breaking anything. Worse for training, the free credit is a per-task constant, so all
eight rollouts of a group receive exactly it and the group's reward variance is zero.
Excluding it makes a do-nothing rollout score 0.0, which is what it earned.

AppWorld's own headline metric (task goal completion) is all-or-nothing; it is recorded
alongside as ``passed`` so reported success rates stay the official ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .verifier import VerifierOutcome

# AppWorld's own definition of solving a task: every requirement test passes.
PASS_REQUIRES_ALL_TESTS = True


def _tracker_of(audit_data: dict[str, Any] | None) -> dict[str, Any]:
    appworld = (audit_data or {}).get("appworld") or {}
    tracker = appworld.get("tracker")
    if not isinstance(tracker, dict):
        raise ValueError(
            "audit_data['appworld']['tracker'] is missing; the environment must run "
            "/evaluate during _finalise"
        )
    return tracker


def grade_rollout(
    *,
    task: Any = None,
    task_id: str,
    task_dir: str | Path | None = None,
    tasks_dir: str | Path | None = None,
    messages: Any = None,
    dispatches: Any = None,
    audit_data: dict[str, Any] | None = None,
    env_snapshot: Any = None,
    judge: Any = None,
    judge_mode: str = "disabled_deterministic_evaluator",
) -> VerifierOutcome:
    """Score one AppWorld rollout from the tracker the environment already collected."""
    try:
        tracker = _tracker_of(audit_data)
    except Exception as exc:  # noqa: BLE001 - an absent tracker is an infra fault
        return VerifierOutcome(
            score=None,
            passed=None,
            status="grader_failure",
            components={"benchmark": "appworld", "task_id": task_id},
            judge_calls=[],
            error=f"{type(exc).__name__}: {exc}",
        )

    passes = tracker.get("passes") or []
    failures = tracker.get("failures") or []
    num_tests = tracker.get("num_tests")
    if not isinstance(num_tests, int) or num_tests <= 0:
        num_tests = len(passes) + len(failures)

    def _is_free(entry: Any) -> bool:
        return isinstance(entry, dict) and entry.get("label") == "no_op_pass"

    earned_passes = [entry for entry in passes if not _is_free(entry)]
    free_passes = [entry for entry in passes if _is_free(entry)]
    free_failures = [entry for entry in failures if _is_free(entry)]
    # Denominator: only the tests an empty agent would fail. A task's free tests are a
    # fixed property of the task, so leaving them in both terms would add the same
    # constant to every rollout of a group.
    scored_tests = num_tests - len(free_passes) - len(free_failures)

    if scored_tests <= 0:
        # Nothing here distinguishes doing the task from doing nothing, so this task
        # cannot measure the policy. That is a benchmark property, not a policy failure,
        # and must not become a reward of zero.
        return VerifierOutcome(
            score=None,
            passed=None,
            status="grader_failure",
            components={
                "benchmark": "appworld",
                "task_id": task_id,
                "num_tests": num_tests,
                "no_op_pass_tests": len(free_passes) + len(free_failures),
            },
            judge_calls=[],
            error="every AppWorld test for this task passes for a do-nothing agent",
        )

    pass_count = len(earned_passes)
    score = round(pass_count / scored_tests, 6)
    official_success = bool(tracker.get("success"))
    appworld_block = (audit_data or {}).get("appworld") or {}

    return VerifierOutcome(
        score=score,
        passed=official_success if PASS_REQUIRES_ALL_TESTS else score >= 0.75,
        status="scored",
        components={
            "benchmark": "appworld",
            "task_id": task_id,
            "grader": "appworld.evaluator (deterministic unit tests)",
            "judge_mode": judge_mode,
            "reward_definition": "earned_passes / tests_a_do_nothing_agent_would_fail",
            "pass_count": pass_count,
            "fail_count": len(failures),
            "num_tests": num_tests,
            "scored_tests": scored_tests,
            # Free credit, reported but never scored: a do-nothing rollout gets 0.0.
            "no_op_pass_tests": len(free_passes) + len(free_failures),
            "raw_pass_ratio": round(len(passes) / num_tests, 6) if num_tests else None,
            # The official metric, kept separate from the training signal.
            "task_goal_completion": official_success,
            "difficulty": tracker.get("difficulty"),
            "scenario_id": str(task_id).rsplit("_", 1)[0],
            "task_completed_signal": appworld_block.get("task_completed"),
            "interactions": appworld_block.get("interactions"),
            "failed_requirements": [
                str(failure.get("requirement"))
                for failure in failures
                if isinstance(failure, dict)
            ][:20],
            "no_op_labels": sorted(
                {
                    str(entry.get("label"))
                    for entry in list(passes) + list(failures)
                    if isinstance(entry, dict) and entry.get("label")
                }
            ),
        },
        judge_calls=[],
        error=None,
    )


__all__ = ["grade_rollout"]
