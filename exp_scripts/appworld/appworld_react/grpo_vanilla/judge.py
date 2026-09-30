"""Stand-in for grpo_vanilla.judge: verifier.py imports only JudgeFailure from it (same body).
AppWorld grading never calls a judge."""


class JudgeFailure(RuntimeError):
    """The judge could not produce a valid score. Infrastructure failure."""
