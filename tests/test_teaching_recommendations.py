from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tailorlearn.analytics import AnalyticsService
from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.models import ProviderProfile


def test_teaching_draft_rejects_blank_summary() -> None:
    """A persisted recommendation must always have a teacher-readable summary."""
    from tailorlearn.models import TeachingRecommendationDraft

    with pytest.raises(ValidationError):
        TeachingRecommendationDraft.model_validate({"summary": " "})


def test_repository_initialization_creates_teaching_recommendation_storage(
    tmp_path: Path,
) -> None:
    """Recommendation drafts need their own durable, queryable storage."""
    repository = Repository(tmp_path / "recommendations.sqlite3")
    repository.initialize()

    with repository.connect() as connection:
        table_names = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert "ai_teaching_recommendations" in table_names


def _services(tmp_path: Path) -> tuple[Repository, LearningDataService, str]:
    repository = Repository(tmp_path / "recommendations.sqlite3")
    repository.initialize()
    learning = LearningDataService(repository, tmp_path)
    dataset = learning.create_dataset("教学建议测试数据")
    return repository, learning, dataset["id"]


def _seed_class_scope(learning: LearningDataService, dataset_id: str) -> None:
    for index in range(1, 7):
        learning.create_record(
            dataset_id,
            "student",
            {"student_code": f"D{index:02d}", "class_code": "DC01", "grade": 7},
        )
    learning.create_record(
        dataset_id,
        "requirement",
        {
            "theme_code": "T01",
            "theme_name": "代数式",
            "theme_goal": "理解代数式",
            "indicator_code": "T01I1",
            "indicator_name": "列代数式",
            "standard_ref": "CR01",
            "standard_page": "62/55",
            "textbook_page": "68-109",
        },
    )
    for index in range(1, 5):
        learning.create_record(
            dataset_id,
            "material",
            {
                "material_code": f"Q{index:03d}",
                "theme_code": "T01",
                "material_type": "题目",
                "question_type": "填空",
                "content": f"题目 {index}",
                "full_score": 1,
                "indicator_codes": "T01I1",
                "source": "题目与材料.csv",
            },
        )
    learning.create_record(
        dataset_id,
        "node",
        {
            "node_code": "N01",
            "test_date": "2026-01-20",
            "taught_topics": "T01",
            "description": "第一次测验",
            "available_materials": "Q001、Q002、Q003、Q004",
        },
    )


def _answer(
    learning: LearningDataService,
    dataset_id: str,
    *,
    student: str,
    material: str,
    score: int | None,
    status: str = "已评分",
) -> None:
    suffix = f"{student}_{material}_{status}"
    learning.create_record(
        dataset_id,
        "answer",
        {
            "record_code": f"R_{suffix}",
            "student_code": student,
            "material_code": material,
            "theme_code": "T01",
            "node_code": "N01",
            "answer_event_code": f"E_{suffix}",
            "revision": 1,
            "submitted_at": "2026-01-20T09:00:00",
            "grading_status": status,
            "score": score,
            "full_score": 1,
            "final_answer": "不得泄露的原始作答",
        },
    )


def test_evidence_pack_groups_only_sufficient_student_and_class_evidence(
    tmp_path: Path,
) -> None:
    """Insufficient records must lead to observation, never negative grouping."""
    repository, learning, dataset_id = _services(tmp_path)
    _seed_class_scope(learning, dataset_id)
    for student in ("D01", "D03", "D04", "D05"):
        for material in ("Q001", "Q002", "Q003"):
            _answer(learning, dataset_id, student=student, material=material, score=1)
    for material in ("Q001", "Q002", "Q003"):
        _answer(learning, dataset_id, student="D02", material=material, score=0)
    _answer(learning, dataset_id, student="D06", material="Q001", score=1)
    _answer(learning, dataset_id, student="D06", material="Q002", score=None, status="待评分")
    _answer(learning, dataset_id, student="D06", material="Q003", score=None, status="未作答")

    from tailorlearn.teaching_recommendations import TeachingRecommendationService

    service = TeachingRecommendationService(repository, AnalyticsService(repository))
    pack = service.build_evidence_pack(dataset_id, "DC01", "N01")
    groups = {(item["theme_code"], item["group"]): item["student_codes"] for item in pack["groups"]}

    assert groups[("T01", "observed_higher")] == ["D01", "D03", "D04", "D05"]
    assert groups[("T01", "needs_attention")] == ["D02"]
    assert groups[("T01", "needs_observation")] == ["D06"]
    assert pack["assessment_blueprint"] == [
        {
            "theme_code": "T01",
            "indicator_code": "T01I1",
            "available_material_codes": ["Q001", "Q002", "Q003", "Q004"],
            "valid_answer_count": 16,
            "pending_count": 1,
            "unanswered_count": 1,
            "coverage_status": "observed",
        }
    ]
    assert "不得泄露的原始作答" not in str(pack)


def test_case_bundle_exposes_only_eligible_candidate_materials(tmp_path: Path) -> None:
    """A class export must not offer a student a material they already answered."""
    repository, learning, dataset_id = _services(tmp_path)
    _seed_class_scope(learning, dataset_id)
    for student in ("D01", "D02", "D03", "D04", "D05"):
        for material in ("Q001", "Q002"):
            _answer(learning, dataset_id, student=student, material=material, score=1)

    from tailorlearn.teaching_recommendations import TeachingRecommendationService

    service = TeachingRecommendationService(repository, AnalyticsService(repository))
    bundle = service.export_case_bundle(dataset_id, "DC01", "N01")
    candidate = next(
        item
        for item in bundle["evidence_pack"]["candidate_materials"]
        if item["material_code"] == "Q001"
    )

    assert candidate["eligible_student_codes"] == ["D06"]
    assert "不得泄露的原始作答" not in str(bundle)


class _RecommendationClient:
    def draft_teaching_recommendation(
        self,
        evidence_pack: dict[str, object],
    ) -> tuple[dict[str, object], str]:
        return (
            {
                "summary": "基于当前充分证据，建议分层组织代数式练习。",
                "observations": [
                    {
                        "theme_code": "T01",
                        "statement": "当前主题已有充分的已观察表现。",
                        "indicator_codes": ["T01I1"],
                    }
                ],
                "information_gaps": [],
                "group_activities": [
                    {
                        "theme_code": "T01",
                        "group": "needs_attention",
                        "activity": "安排同指标的分步练习。",
                        "student_codes": ["D02"],
                    }
                ],
                "material_suggestions": [
                    {
                        "theme_code": "T01",
                        "indicator_code": "T01I1",
                        "material_code": "Q004",
                        "student_codes": ["D02"],
                        "reason": "作为补充观察材料。",
                    }
                ],
                "teacher_actions": ["教师审核后安排分层活动。"],
                "post_lesson_checks": ["记录补充观察结果。"],
            },
            "{\"summary\": \"raw\"}",
        )


def _profile() -> ProviderProfile:
    return ProviderProfile(
        id="provider_test",
        vendor_name="Test",
        base_url="https://example.test/v1",
        model="test-model",
        api_key="test-key",
    )


def test_recommendation_is_reviewable_and_stale_after_learning_data_changes(
    tmp_path: Path,
) -> None:
    """Accepted teaching guidance must not remain current after new evidence arrives."""
    repository, learning, dataset_id = _services(tmp_path)
    _seed_class_scope(learning, dataset_id)
    for student in ("D01", "D03", "D04", "D05"):
        for material in ("Q001", "Q002", "Q003"):
            _answer(learning, dataset_id, student=student, material=material, score=1)
    for material in ("Q001", "Q002", "Q003"):
        _answer(learning, dataset_id, student="D02", material=material, score=0)

    from tailorlearn.teaching_recommendations import TeachingRecommendationService

    service = TeachingRecommendationService(repository, AnalyticsService(repository))
    created = service.create_recommendation(
        dataset_id,
        "DC01",
        "N01",
        _profile(),
        _RecommendationClient(),
    )
    accepted = service.review_recommendation(
        created["id"],
        status="accepted",
        reviewed_summary="教师采用分层练习。",
        review_note="先小范围试行。",
    )
    _answer(learning, dataset_id, student="D06", material="Q004", score=1)

    assert accepted["status"] == "accepted"
    assert service.refresh_stale_recommendations(dataset_id) == 1
    assert service.get_latest_recommendation(dataset_id, "DC01", "N01")["status"] == "stale"
