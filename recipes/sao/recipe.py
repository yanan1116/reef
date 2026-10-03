"""Single-Rollout Asynchronous Optimization recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recipes.sao.processor import SAOProcessor
from recipes.sao.report import SAOReport
from reef.core.reports import ReportBase
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.train.algos import StepScheduling


@dataclass(frozen=True, kw_only=True)
class SAORecipe(WeightTrainingRecipe):
    """Single-Rollout Asynchronous Optimization (arXiv:2607.07508) on reef.

    "Single rollout" is one rollout per prompt: there is no comparison group
    and no slowest-sample barrier, so each scored rollout is accepted on its
    own as it lands. ``batch_size`` of them, from ``batch_size`` different
    prompts, form one optimizer step; the paper trains with 128 (§4.1). The
    value model is what makes a single sample per prompt usable, and it needs
    that many samples per step to learn, so ``batch_size=1`` (one step per
    rollout) is a smoke setting, not the paper's estimator. The DIS ratio
    needs the rollout log-probabilities as its behaviour proxy, so SAO requires
    an inference backend that attaches engine-native tensors
    (``reef.inference_handler_factory``); reef never re-tokenizes a rollout to
    reconstruct them.

    Objective settings such as the clipping bounds, actor/critic cadence, and GAE
    parameters belong to the training backend. For Slime they are configured by
    ``training.options``; this recipe only owns Reef-side batching and
    checkpoint cadence.

    ``batch_size`` must equal the Slime driver's ``--global-batch-size``: each
    rollout sample is its own DP unit.

    ``privileged_value`` turns the critic into a privileged value function
    (arXiv:2608.16739): every report must carry a ``critic_context``, which
    the served model's tokenizer (``tokenizer_path``) renders as a system
    turn that only the critic reads, before the sample. The actor's sample,
    the DIS ratio, the loss and the GAE are unchanged; off, a report carrying
    a context is rejected, so the flag alone decides which arm a run is.
    """

    name: str = "sao"
    batch_size: int = config_field(128, env="REEF_SAO_BATCH_SIZE")
    privileged_value: bool = config_field(False)
    tokenizer_path: str = config_field("")

    @property
    def report_type(self) -> type[ReportBase]:
        return SAOReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="sao",
            processor=SAOProcessor,
            # Each rollout is its own DP unit; the backend's configured step size applies.
            scheduling=StepScheduling(unit="sample"),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.privileged_value and not self.tokenizer_path.strip():
            raise ValueError(
                "privileged_value needs tokenizer_path: the served model's tokenizer renders the critic context"
            )

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError(
                "SAO objective options are backend-owned; configure the Slime implementation with training.options"
            )
