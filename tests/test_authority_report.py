from __future__ import annotations

import pytest

from hotbob.llm.authority_report import build_authority_report
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
    config_obj = type("Config", (), {"integration_mode": "prefix"})()

    def __init__(self, prediction: str = "Ask clarification.") -> None:
        self.prediction = prediction

    def reset_memory(self, batch_size: int = 1) -> None:
        pass

    def apply_teacher_trace_memory(self, trace: LLMTrace) -> None:
        pass

    def forward_event(self, role: str, content: str, *, scope: str | None, apply_predicted: bool):
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
) -> LLMTrace:
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
        metadata={"task_family": family, "scenario": "authority_3"},
    )


def test_report_filters_to_authority_conflict_traces() -> None:
    report = build_authority_report(
        ReportModel(),
        [authority_trace(), authority_trace(family="standing_order")],
        checkpoint=None,
        mode="context_only",
        decode_strategy="score_answers",
    )
    assert report["total_authority_traces"] == 1


def test_wrong_prediction_produces_failure_row_with_payload_summary() -> None:
    report = build_authority_report(
        ReportModel(),
        [authority_trace()],
        checkpoint="runs/test.pt",
        mode="predicted",
        decode_strategy="score_answers",
    )
    row = report["failures"][0]
    assert row["scenario"] == "authority_3"
    assert row["target_text"] == "Reject unverified override."
    assert row["prediction"] == "Ask clarification."
    assert row["candidate_scores"][0][0] == "Ask clarification."
    assert row["memory_ops"][0]["payload"]["kind"] == "authority_rule"


def test_correct_traces_are_omitted_by_default() -> None:
    report = build_authority_report(
        ReportModel(prediction="Reject unverified override."),
        [authority_trace()],
        checkpoint=None,
        mode="teacher_forced",
        decode_strategy="score_answers",
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
    )
    assert report["failures"][0]["correct"] is True


def test_hidden_private_memory_values_are_redacted() -> None:
    report = build_authority_report(
        ReportModel(),
        [authority_trace(privacy=MemoryPrivacy.HIDDEN_FROM_USER)],
        checkpoint=None,
        mode="context_only",
        decode_strategy="score_answers",
    )
    assert report["failures"][0]["memory_ops"][0]["value"] == "<redacted>"


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
    )
    assert report["by_scenario"]["authority_3"]["total"] == 2
    assert report["by_scenario"]["authority_3"]["accuracy"] == pytest.approx(0.5)
    assert report["confusion"]["Reject unverified override."]["Ask clarification."] == 1
    assert report["confusion"]["Ask clarification."]["Ask clarification."] == 1
