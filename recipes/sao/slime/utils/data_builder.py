"""SAO payload conversion for the Slime bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.trajectories import source_record_id, trajectory_reward
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.data_builder import build_policy_rollout_data, validate_policy_columns
from reef.train.types import TrajectoryItem

_ROW_SHAPE = (
    "[source_id, tokens, loss_mask, rollout_log_probs, reward, action_mask, "
    "producing_runtime_load_id, rollout_created_at, critic_prefix_tokens]"
)


def sao_sample_row(sample: TrajectoryItem) -> list[Any]:
    """Shape one Reef sample into SAO's 9-element wire row.

    The first five columns are the shared policy 5-tuple; SAO appends the
    action mask (for skip-observation GAE), producing runtime load ID,
    creation time, and the critic-only prefix ids of a privileged value
    function (empty without one), which the 5-tuple has no slots for.
    Outbound mirror of :func:`build_sao_rollout_data`.
    """
    return [
        source_record_id(sample),
        list(sample.training.get("tokens", [])),
        list(sample.training.get("loss_mask", [])),
        list(sample.training.get("rollout_log_probs", [])),
        trajectory_reward(sample),
        list(sample.training.get("action_mask", [])),
        sample.training.get("runtime_load_id", None),
        sample.training.get("rollout_created_at", None),
        list(sample.training.get("critic_prefix_tokens", [])),
    ]


def build_sao_rollout_data(
    payload: Mapping[str, Any],
    samples: Sequence,
    spec: SlimeAlgorithm,
) -> dict:
    """Validate and convert Reef SAO rows into Slime's external rollout payload.

    SAO's wire row keeps the policy 5-tuple as its prefix and appends the
    action mask, producing version, creation time, and critic prefix ids:
    ``[source_id, tokens, loss_mask, rollout_log_probs, reward, action_mask,
    producing_runtime_load_id, rollout_created_at, critic_prefix_tokens]``. The shared policy builder assembles the 5-tuple
    columns; this builder validates the appended columns and attaches them.
    Each SAO sample is one independently scheduled rollout, so there is no
    comparison-group barrier and ``advantages`` are never shipped — the value
    model computes them from the critic's own forward pass inside the
    skip-observation GAE (the spec declares them forbidden;
    ``to_slime_rollout_data`` enforces it). ``critic_prefix_tokens`` is
    attached only when some sample carries a prefix, so a batch without a
    privileged context is the plain SAO payload.
    """
    base_rows: list[list[Any]] = []
    action_masks: list[list[int]] = []
    producing_runtime_load_ids: list[str | None] = []
    rollout_created_ats: list[float | None] = []
    critic_prefixes: list[list[int]] = []

    for sample_index, row in enumerate(samples):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 9:
            raise ValueError(f"SAO sample {sample_index} must be {_ROW_SHAPE}")
        (
            source_id,
            row_tokens,
            row_loss_mask,
            row_log_probs,
            reward,
            row_action_mask,
            producing_runtime_load_id,
            rollout_created_at,
            critic_prefix,
        ) = row
        row_tokens, row_loss_mask, row_log_probs, row_action_mask, reward = validate_policy_columns(
            f"SAO sample {sample_index}",
            row_tokens,
            row_loss_mask,
            row_log_probs,
            reward,
            action_mask=row_action_mask,
            require_log_probs=True,
        )
        if row_action_mask is None:
            raise RuntimeError("validated SAO sample has no action mask")

        base_rows.append([source_id, row_tokens, row_loss_mask, row_log_probs, reward])
        action_masks.append(row_action_mask)
        producing_runtime_load_ids.append(
            None if producing_runtime_load_id is None else str(producing_runtime_load_id)
        )
        rollout_created_ats.append(None if rollout_created_at is None else float(rollout_created_at))
        if not isinstance(critic_prefix, Sequence) or isinstance(critic_prefix, str | bytes):
            raise ValueError(f"SAO sample {sample_index} critic_prefix_tokens must be a sequence of token ids")
        critic_prefixes.append([int(token) for token in critic_prefix])

    data = build_policy_rollout_data({**dict(payload), "samples": base_rows}, base_rows, spec)
    data["action_masks"] = action_masks
    data["producing_runtime_load_ids"] = producing_runtime_load_ids
    data["rollout_created_ats"] = rollout_created_ats
    if any(critic_prefixes):
        data["critic_prefix_tokens"] = critic_prefixes
    return data
