"""SAO's report contract: one scored rollout, plus the privileged text only its critic reads."""

from dataclasses import dataclass

from reef.core.reports import ScoredRolloutReport


@dataclass(frozen=True)
class SAOReport(ScoredRolloutReport):
    """A scored rollout and, for a privileged value function, the critic's context.

    ``critic_context`` is text the value model conditions on and the policy
    never sees (Le Critique: Privileged Value Functions, arXiv:2608.16739),
    such as the task's reference answer. It must not depend on the rollout's
    own actions. Empty means a plain SAO report; a recipe with
    ``privileged_value`` requires it on every report.
    """

    critic_context: str = ""


__all__ = ["SAOReport"]
