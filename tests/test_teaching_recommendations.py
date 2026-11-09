from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tailorlearn.analytics import AnalyticsService
from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService


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
