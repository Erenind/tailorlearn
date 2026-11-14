from __future__ import annotations

from pathlib import Path

from tailorlearn.analytics import AnalyticsService
from tailorlearn.db import Repository
from tailorlearn.home_dashboard import HomeDashboardService
from tailorlearn.learning_data import LearningDataService
from tailorlearn.student_diagnosis import StudentDiagnosisService
from tailorlearn.teacher_workbench import TeacherWorkbenchService


def _services(tmp_path: Path) -> tuple[LearningDataService, str, HomeDashboardService]:
    repository = Repository(tmp_path / "dashboard.sqlite3")
    repository.initialize()
    learning = LearningDataService(repository, tmp_path)
    analytics = AnalyticsService(repository)
    dataset = learning.create_dataset("首页数据集")
    workbench = TeacherWorkbenchService(
        repository,
        learning,
        analytics,
        StudentDiagnosisService(repository, analytics),
    )
    return learning, dataset["id"], HomeDashboardService(learning, analytics, workbench)


def _record(
    learning: LearningDataService,
    dataset_id: str,
    data_type: str,
    values: dict[str, object],
) -> None:
    learning.create_record(dataset_id, data_type, values)


def test_home_dashboard_reports_data_states_without_mastery_claims(tmp_path: Path) -> None:
    learning, dataset_id, dashboard = _services(tmp_path)
    for index in range(1, 6):
        _record(
            learning,
            dataset_id,
            "student",
            {"student_code": f"D{index:02}", "class_code": "DC01", "grade": 7},
        )
    _record(
        learning,
        dataset_id,
        "requirement",
        {
            "theme_code": "T01",
            "theme_name": "代数式",
            "theme_goal": "理解代数式",
            "indicator_code": "T01I1",
            "indicator_name": "列代数式",
            "standard_ref": "CR01",
            "standard_page": "1",
            "textbook_page": "1",
        },
    )
    _record(
        learning,
        dataset_id,
        "material",
        {
            "material_code": "Q01",
            "theme_code": "T01",
            "material_type": "题目",
            "question_type": "填空",
            "content": "题目",
            "full_score": 1,
            "indicator_codes": "T01I1",
            "source": "测试",
        },
    )
    _record(
        learning,
        dataset_id,
        "node",
        {
            "node_code": "N01",
            "test_date": "2026-01-20",
            "taught_topics": "T01",
            "description": "测试",
            "available_materials": "Q01",
        },
    )
    for index in range(1, 6):
        for attempt in range(2):
            _record(
                learning,
                dataset_id,
                "answer",
                {
                    "record_code": f"R{index}{attempt}",
                    "student_code": f"D{index:02}",
                    "material_code": "Q01",
                    "theme_code": "T01",
                    "node_code": "N01",
                    "answer_event_code": f"E{index}{attempt}",
                    "revision": 1,
                    "submitted_at": "2026-01-20T09:00:00",
                    "grading_status": "已评分",
                    "score": 1,
                    "full_score": 1,
                    "final_answer": "敏感作答不应展示",
                },
            )
    _record(
        learning,
        dataset_id,
        "answer",
        {
            "record_code": "R_PENDING",
            "student_code": "D01",
            "material_code": "Q01",
            "theme_code": "T01",
            "node_code": "N01",
            "answer_event_code": "E_PENDING",
            "revision": 1,
            "submitted_at": "2026-01-20T09:00:00",
            "grading_status": "待评分",
            "score": None,
            "full_score": 1,
            "final_answer": "也不应展示",
        },
    )

    summary = dashboard.build(dataset_id)

    assert summary["metrics"] == {
        "student_count": 5,
        "valid_scored_answers": 10,
        "data_processing_todos": 1,
        "follow_up_todos": 0,
    }
    assert summary["evidence_snapshot"] == [
        {"label": "证据充分", "count": 1},
        {"label": "待补充观察", "count": 0},
    ]
    assert "未掌握" not in str(summary)
    assert "敏感作答" not in str(summary)
