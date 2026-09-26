"""SAO recipe that trains one multi-turn FinQA episode as one sample.

A FinQA episode is several model calls (the ReAct turns), each with its own
receipt. The driver reports the episode's score once, referencing every turn's
receipt in order; Reef's SampleAssembly then builds one linear sample from them
(make_multi_turn_policy_trajectory): each turn's response tokens are trained
(loss_mask 1) and the tool outputs and next-turn prompt tokens in between are
context (loss_mask 0). SAO's skip-observation GAE runs over that mask, which is
the paper's treatment of observations.

The stock SAORecipe does not declare the assembly switches, and recipe config
rejects keys a recipe does not consume, so this subclass declares them; nothing
else changes (same objective, processor, batching, and checkpoint cadence).
The assembly requires every turn of an episode to come from one engine weight
version; stream_finqa.py drops episodes that straddle a weight publication
before reporting them.

Loaded inside the stack container as
  recipe.implementation: sao_multiturn:MultiTurnSAORecipe
with exp_scripts/finqa on PYTHONPATH (scripts/start_stack.sh EXTRA_PYTHONPATH).
"""

from __future__ import annotations

from dataclasses import dataclass

from recipes.sao.recipe import SAORecipe
from reef.recipe.config_fields import config_field


@dataclass(frozen=True, kw_only=True)
class MultiTurnSAORecipe(SAORecipe):
    """SAORecipe plus the reported-feedback processor's multi-turn assembly settings."""

    accept_multi_turn_policy_samples: bool = config_field(True)
    # Drift allowed when a turn's prompt re-renders the previous response differently
    # (tool-call formatting): realigned as masked context, never as trained tokens.
    realign_threshold: int = config_field(1024)
    # Qwen3-4B-Instruct-2507 has no thinking scaffold, so no drift before a response.
    scaffold_tolerance: int = config_field(0)
