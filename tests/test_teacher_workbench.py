from __future__ import annotations

import json
from pathlib import Path

from tailorlearn.analytics import AnalyticsService
from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.student_diagnosis import StudentDiagnosisService
from tailorlearn.teaching_recommendations import TeachingRecommendationService


def _services(tmp_path: Path) -> tuple[
    Repository,
    LearningDataService,
    AnalyticsService,
    str,
]:
    repository = Repository(tmp_path / "workbench.sqlite3")
    repository.initialize()
    learning = LearningDataService(repository, tmp_path)
    dataset = learning.create_dataset("工作台测试数据")
    return repository, learning, AnalyticsService(repository), dataset["id"]


def _seed_minimum_scope(learning: LearningDataService, dataset_id: str) -> None:
    learning.create_record(
        dataset_id,
        "student",
        {"student_code": "D01", "class_code": "DC01", "grade": 7},
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
            "standard_page": "1/1",
            "textbook_page": "1",
        },
    )
    learning.create_record(
        dataset_id,
        "material",
        {
            "material_code": "Q001",
            "theme_code": "T01",
            "material_type": "题目",
            "question_type": "填空",
            "content": "题目",
            "full_score": 1,
            "indicator_codes": "T01I1",
            "source": "测试",
        },
    )
    learning.create_record(
        dataset_id,
        "node",
        {
            "node_code": "N01",
            "test_date": "2026-01-20",
            "taught_topics": "T01",
            "description": "测试",
            "available_materials": "Q001",
        },
    )


def _answer(
    learning: LearningDataService,
    dataset_id: str,
    *,
    record_code: str,
    event_code: str,
    status: str,
    score: int | None,
    final_answer: str,
) -> None:
    learning.create_record(
        dataset_id,
        "answer",
        {
            "record_code": record_code,
            "student_code": "D01",
            "material_code": "Q001",
            "theme_code": "T01",
            "node_code": "N01",
            "answer_event_code": event_code,
            "revision": 1,
            "submitted_at": "2026-01-20T09:00:00",
            "grading_status": status,
            "score": score,
            "full_score": 1,
            "final_answer": final_answer,
        },
    )


def test_workbench_aggregates_actionable_items_without_exposing_answers(
    tmp_path: Path,
) -> None:
    """Pending and invalid records are actionable, never mastery conclusions."""
    repository, learning, analytics, dataset_id = _services(tmp_path)
    _seed_minimum_scope(learning, dataset_id)
    _answer(
        learning,
        dataset_id,
        record_code="R_PENDING",
        event_code="E_PENDING",
        status="待评分",
        score=None,
        final_answer="不得出现在工作台",
    )
    _answer(
        learning,
        dataset_id,
        record_code="R_INVALID",
        event_code="E_INVALID",
        status="已评分",
        score=2,
        final_answer="也不得出现在工作台",
    )

    from tailorlearn.teacher_workbench import TeacherWorkbenchService

    service = TeacherWorkbenchService(
        repository,
        learning,
        analytics,
        StudentDiagnosisService(repository, analytics),
        TeachingRecommendationService(repository, analytics),
    )
    board = service.build_todos(dataset_id)

    categories = {item["category"] for item in board["todos"]}
    assert {"data_processing", "supplemental_observation"} <= categories
    assert board["summary"]["data_processing"] >= 2
    assert board["summary"]["supplemental_observation"] == 1
    assert all("未掌握" not in item["reason"] for item in board["todos"])
    assert "不得出现在工作台" not in str(board)
    assert "也不得出现在工作台" not in str(board)


def test_workbench_marks_latest_ai_drafts_for_review_then_refreshes_them(
    tmp_path: Path,
) -> None:
    repository, learning, analytics, dataset_id = _services(tmp_path)
    _seed_minimum_scope(learning, dataset_id)
    with repository.connect() as connection:
        cutoff = connection.execute(
            "SELECT MAX(id) FROM learning_change_log WHERE dataset_id=?",
            (dataset_id,),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO ai_diagnosis_jobs
            (id, dataset_id, class_code, node_code, change_log_cutoff,
             provider_id, provider_vendor, provider_base_url, provider_model,
             status, total, created_at, updated_at)
            VALUES ('job1', ?, 'DC01', 'N01', ?, 'provider', 'test',
                    'https://example.test', 'test-model', 'completed', 1,
                    '2026-01-20T09:00:00', '2026-01-20T09:00:00')
            """,
            (dataset_id, cutoff),
        )
        connection.execute(
            """
            INSERT INTO ai_student_diagnoses
            (id, job_id, dataset_id, student_code, node_code, change_log_cutoff,
             evidence_pack_json, response_json, raw_response, provider_id,
             provider_vendor, provider_base_url, provider_model, status,
             created_at, updated_at)
            VALUES ('diagnosis1', 'job1', ?, 'D01', 'N01', ?, '{}',
                    '{"summary": "草稿"}', '{}', 'provider', 'test',
                    'https://example.test', 'test-model', 'needs_review',
                    '2026-01-20T09:00:00', '2026-01-20T09:00:00')
            """,
            (dataset_id, cutoff),
        )
        connection.execute(
            """
            INSERT INTO ai_teaching_recommendations
            (id, dataset_id, class_code, node_code, change_log_cutoff,
             evidence_pack_json, response_json, raw_response, provider_id,
             provider_vendor, provider_base_url, provider_model, prompt_version,
             status, created_at, updated_at)
            VALUES ('recommendation1', ?, 'DC01', 'N01', ?, '{}',
                    '{"summary": "草稿"}', '{}', 'provider', 'test',
                    'https://example.test', 'test-model', 'class-teaching-v1',
                    'needs_review', '2026-01-20T09:00:00', '2026-01-20T09:00:00')
            """,
            (dataset_id, cutoff),
        )
    service = __import__(
        "tailorlearn.teacher_workbench", fromlist=["TeacherWorkbenchService"]
    ).TeacherWorkbenchService(
        repository,
        learning,
        analytics,
        StudentDiagnosisService(repository, analytics),
        TeachingRecommendationService(repository, analytics),
    )

    initial = service.build_todos(dataset_id)
    assert initial["summary"]["ai_review"] == 2
    assert initial["summary"]["needs_update"] == 0

    _answer(
        learning,
        dataset_id,
        record_code="R_AFTER_DRAFT",
        event_code="E_AFTER_DRAFT",
        status="待评分",
        score=None,
        final_answer="不应泄露",
    )
    refreshed = service.build_todos(dataset_id)
    assert refreshed["summary"]["ai_review"] == 0
    assert refreshed["summary"]["needs_update"] == 2

