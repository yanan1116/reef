"""SAO reported-feedback processor: one completed rollout, one training unit."""

from __future__ import annotations

from recipes.sao.report import SAOReport
from reef.core.records_types import AgentRecord
from reef.train.processors.common import make_policy_trajectory
from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem


def make_sao_sample(item: AgentRecord, reward: float) -> TrajectoryItem:
    """Convert inference data and its evaluated reward into an SAO sample.

    ``make_policy_trajectory`` captures the shared ATIF training fields, so SAO and the
    group-relative processors resolve every shared field identically —
    including the ``runtime_load_id`` fallback chain the durable runtime needs
    to identify a training job's producing version. SAO then fills the two
    fields that path leaves at their defaults: ``action_mask`` (read from
    ``response.training`` first, the top-level payload second) and
    ``rollout_created_at``, for the backend's queue-age metric.
    """
    base = make_policy_trajectory(item, reward)
    payload = item.payload
    response = payload.get("response", {})
    training = response.get("training", {}) if isinstance(response, dict) else {}
    action_mask = training.get("action_mask", payload.get("action_mask", ())) if isinstance(training, dict) else ()
    return base.with_training(action_mask=[int(value) for value in action_mask], rollout_created_at=item.created_at)


class SAOProcessor(ReportedFeedbackProcessor):
    """Turn scored rollouts into independently-scheduled SAO samples.

    Single-Rollout Asynchronous Optimization ships each completed rollout on
    its own — no comparison group, no slowest-sample barrier. The dispatcher
    collects ``batch_size`` accepted rollouts (from as many prompts) into one
    training step; the recipe default is the paper's 128, and 1 trains once
    per accepted rollout for smoke runs.

    SAO reuses ``TrajectoryItem`` / ``TrainingBatch`` and fills ``action_mask``
    and ``rollout_created_at``. The training backend validates required
    tensors; malformed training input fails explicitly.
    """

    output_schema = TrainingBatch
    exclusive_sources = True

    def __init__(self, context: ProcessorContext) -> None:
        config = context.config
        self._assembly = SampleAssembly.from_config(context, make_sample=make_sao_sample)
        self._privileged_value = bool(config.get("privileged_value", False))
        self._tokenizer = None
        if self._privileged_value:
            tokenizer_path = str(config.get("tokenizer_path", "")).strip()
            if not tokenizer_path:
                raise ValueError(
                    "privileged_value needs tokenizer_path: the served model's tokenizer renders the critic context"
                )
            # transformers belongs to the training environment; the service never renders a prompt.
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        super().__init__(context)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        sample = self._assembly.build(context, context.require_score())
        if not sample.training.get("action_mask", []):
            sample = sample.with_training(action_mask=sample.training.get("loss_mask", []))
        critic_context = self._critic_context(context)
        if critic_context:
            sample = sample.with_training(critic_prefix_tokens=self.critic_prefix_tokens(critic_context))
        return sample

    def _critic_context(self, context: ReportContext) -> str:
        """The report's privileged critic text; required exactly when the recipe has ``privileged_value``."""
        parsed = context.parsed_report
        critic_context = parsed.critic_context if isinstance(parsed, SAOReport) else ""
        report_id = context.report.agent_record_id
        if self._privileged_value and not critic_context.strip():
            raise ValueError(
                f"report {report_id} has no critic_context, but the recipe's privileged_value critic needs one "
                "on every report; send metadata.critic_context or turn privileged_value off"
            )
        if not self._privileged_value and critic_context:
            raise ValueError(
                f"report {report_id} carries a critic_context, but the recipe's privileged_value is off, so the "
                "critic would silently ignore it; turn privileged_value on or stop sending the context"
            )
        return critic_context

    def critic_prefix_tokens(self, critic_context: str) -> list[int]:
        """``critic_context`` as a system turn in the served model's chat template, the critic's sequence prefix."""
        if self._tokenizer is None:
            raise RuntimeError("critic prefix requested without a tokenizer (privileged_value is off)")
        ids = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": critic_context}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        return [int(token) for token in ids]

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:sao:{batch_number}", items)
