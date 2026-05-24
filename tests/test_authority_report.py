from __future__ import annotations

import pytest

from hotbob.llm.authority_report import (
    build_authority_report,
    collect_authority_write_diagnostics,
    compact_authority_report,
)
from hotbob.llm.dataset import LLMTrace, LLMTraceEvent
from hotbob.types import (
    AuthorityLevel,
    AuthorityRulePayload,
    MemoryAuthority,
    MemoryOp,
    MemoryOpName,
    MemoryPayloadKind,
    MemoryPrivacy,
    MemoryType,
    PolicyAction,
)


class ReportModel:
    config_obj = type(
        "Config",
        (),
        {"integration_mode": "prefix", "scope_vocab": {"mission_4": 1, "default": 1}},
    )()

    def __init__(self, prediction: str = "Ask clarification.") -> None:
        self.prediction = prediction

    def reset_memory(self, batch_size: int = 1) -> None:
        pass

    def apply_teacher_trace_memory(self, trace: LLMTrace) -> None:
        pass

    def forward_event(
        self,
        role: str,
        content: str,
        *,
        scope: str | None,
        apply_predicted: bool = False,
    ):
        return {}

    def score_final_candidates_dict(
        self,
        final_prompt: str,
        candidates: list[str],
        *,
        current_scope: str,
        use_memory: bool = True,
    ) -> dict[str, float]:
        scores = {candidate: -10.0 for candidate in candidates}
        scores["Reject unverified override."] = -1.31
        scores["Ask clarification."] = -1.23
        if self.prediction == "Reject unverified override.":
            scores["Reject unverified override."] = -1.0
            scores["Ask clarification."] = -1.5
        return scores


def authority_trace(
    *,
    target: str = "Reject unverified override.",
    privacy: MemoryPrivacy = MemoryPrivacy.VISIBLE,
    family: str = "authority_conflict",
    metadata_overrides: dict[str, object] | None = None,
) -> LLMTrace:
    metadata = {
        "task_family": family,
        "scenario": "authority_3",
        "winning_authority": "captain",
        "losing_authority": "tool_unverified",
        "conflict_action": "ask_clarification",
        "authority_prompt_injection": True,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    return LLMTrace(
        events=[LLMTraceEvent(role="SYSTEM", content="Use captain authority.", scope="mission_4")],
        final_prompt="Answer:",
        target_text=target,
        target_action="REJECT_UNVERIFIED_OVERRIDE",
        current_scope="mission_4",
        expected_memory_ops=[
            MemoryOp(
                op=MemoryOpName.WRITE,
                type=MemoryType.AUTHORITY_RULE,
                key="authority_conflict",
                value="private detail",
                scope="mission_4",
                privacy=privacy,
                authority=MemoryAuthority.USER,
                payload=AuthorityRulePayload(
                    kind=MemoryPayloadKind.AUTHORITY_RULE,
                    subject_key="authority_conflict",
                    winning_authority=AuthorityLevel.CAPTAIN,
                    losing_authority=AuthorityLevel.TOOL_UNVERIFIED,
                    conflict_action=PolicyAction.ASK_CLARIFICATION,
                ),
            )
        ],
        metadata=metadata,
    )


def test_report_filters_to_authority_conflict_traces() -> None:
    report = build_authority_report(
        ReportModel(),
        [authority_trace(), authority_trace(family="standing_order")],
        checkpoint=None,
        mode="context_only",
        decode_strategy="score_answers",
        include_write_diagnostics=False,
        include_read_diagnostics=False,
    )
    assert report["total_authority_traces"] == 1


def test_wrong_prediction_produces_failure_row_with_payload_summary() -> None:
    report = build_authority_report(
        ReportModel(),
        [authority_trace()],
        checkpoint="runs/test.pt",
        mode="predicted",
        decode_strategy="score_answers",
        include_write_diagnostics=False,
        include_read_diagnostics=False,
    )
    row = report["failures"][0]
    assert row["scenario"] == "authority_3"
    assert row["target_text"] == "Reject unverified override."
    assert row["prediction"] == "Ask clarification."
    assert row["candidate_scores"][0][0] == "Ask clarification."
    assert row["memory_ops"][0]["payload"]["kind"] == "authority_rule"
    assert row["target_rank"] == 2
    assert row["target_score"] == pytest.approx(-1.31)
    assert row["prediction_score"] == pytest.approx(-1.23)
    assert row["target_margin"] == pytest.approx(-0.08)
    assert row["winning_authority"] == "captain"
    assert row["losing_authority"] == "tool_unverified"
    assert row["authority_prompt_injection"] is True


def test_correct_traces_are_omitted_by_default() -> None:
    report = build_authority_report(
        ReportModel(prediction="Reject unverified override."),
        [authority_trace()],
        checkpoint=None,
        mode="teacher_forced",
        decode_strategy="score_answers",
        include_write_diagnostics=False,
        include_read_diagnostics=False,
    )
    assert report["accuracy"] == pytest.approx(1.0)
    assert report["failures"] == []


def test_include_correct_includes_correct_authority_traces() -> None:
    report = build_authority_report(
        ReportModel(prediction="Reject unverified override."),
        [authority_trace()],
        checkpoint=None,
        mode="teacher_forced",
        decode_strategy="score_answers",
        include_correct=True,
        include_write_diagnostics=False,
        include_read_diagnostics=False,
    )
    assert report["failures"][0]["correct"] is True


def test_hidden_private_memory_values_are_redacted() -> None:
    report = build_authority_report(
        ReportModel(),
        [authority_trace(privacy=MemoryPrivacy.HIDDEN_FROM_USER)],
        checkpoint=None,
        mode="context_only",
        decode_strategy="score_answers",
        include_write_diagnostics=False,
        include_read_diagnostics=False,
    )
    assert report["failures"][0]["memory_ops"][0]["value"] == "<redacted>"


def test_prediction_distribution_and_prompt_injection_slices() -> None:
    report = build_authority_report(
        ReportModel(),
        [
            authority_trace(),
            authority_trace(
                target="Ask clarification.",
                metadata_overrides={"authority_prompt_injection": False},
            ),
        ],
        checkpoint=None,
        mode="context_only",
        decode_strategy="score_answers",
        include_read_diagnostics=False,
        include_write_diagnostics=False,
    )
    assert report["prediction_distribution"]["Ask clarification."] == pytest.approx(1.0)
    assert report["by_prompt_injection"]["true"]["total"] == 1
    assert report["by_prompt_injection"]["false"]["total"] == 1


def test_compact_report_omits_failure_rows() -> None:
    report = build_authority_report(
        ReportModel(),
        [authority_trace()],
        checkpoint=None,
        mode="context_only",
        decode_strategy="score_answers",
        include_read_diagnostics=False,
        include_write_diagnostics=False,
    )
    compact = compact_authority_report(report)
    assert "failures" not in compact
    assert compact["accuracy"] == report["accuracy"]


def test_write_diagnostics_track_structured_authority_heads() -> None:
    class WriteModel(ReportModel):
        def forward_event(
            self,
            role: str,
            content: str,
            *,
            scope: str | None,
            apply_predicted: bool = False,
        ):
            import torch

            num_levels = 8
            return {
                "op_logits": torch.tensor([[0, 0, 1, 0]], dtype=torch.float32),
                "type_logits": torch.tensor([[0]], dtype=torch.float32),
                "scope_logits": torch.tensor([[0]], dtype=torch.float32),
                "privacy_logits": torch.tensor([[0]], dtype=torch.float32),
                "authority_logits": torch.tensor([[0]], dtype=torch.float32),
                "slot_logits": torch.tensor([[0]], dtype=torch.float32),
                "payload_kind_logits": torch.tensor([[1]], dtype=torch.float32),
                "payload_default_action_logits": torch.zeros(1, 8),
                "payload_winning_authority_level_logits": torch.zeros(1, num_levels),
                "payload_losing_authority_level_logits": torch.zeros(1, num_levels),
            }

    trace = authority_trace()
    diag = collect_authority_write_diagnostics(WriteModel(), trace)
    assert diag["write_events"] >= 1
    assert "payload_winning_authority_level" in diag["by_head"]


def test_by_scenario_accuracy_and_confusion_counts_are_correct() -> None:
    report = build_authority_report(
        ReportModel(),
        [
            authority_trace(),
            authority_trace(target="Ask clarification."),
        ],
        checkpoint=None,
        mode="context_only",
        decode_strategy="score_answers",
        include_read_diagnostics=False,
        include_write_diagnostics=False,
    )
    assert report["by_scenario"]["authority_3"]["total"] == 2
    assert report["by_scenario"]["authority_3"]["accuracy"] == pytest.approx(0.5)
    assert report["target_rank_mean"] == pytest.approx(1.5)
    assert report["mean_target_margin"] == pytest.approx(0.0)
    assert report["by_winning_authority"]["captain"]["total"] == 2
    assert report["by_losing_authority"]["tool_unverified"]["accuracy"] == pytest.approx(0.5)
    assert report["by_target_action"]["REJECT_UNVERIFIED_OVERRIDE"]["total"] == 2
    assert report["confusion"]["Reject unverified override."]["Ask clarification."] == 1
    assert report["confusion"]["Ask clarification."]["Ask clarification."] == 1
