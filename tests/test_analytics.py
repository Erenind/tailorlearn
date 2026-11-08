from __future__ import annotations

from importlib import import_module
from pathlib import Path

from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService


def _services(tmp_path: Path) -> tuple[Repository, LearningDataService, str]:
    repository = Repository(tmp_path / "analytics.sqlite3")
    repository.initialize()
    learning = LearningDataService(repository, tmp_path)
    dataset = learning.create_dataset("分析测试数据")
    return repository, learning, dataset["id"]


def _seed_base(learning: LearningDataService, dataset_id: str) -> None:
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
    record: str,
    event: str,
    material: str,
    revision: int = 1,
    status: str = "已评分",
    score: int | None = 1,
    student: str = "D01",
) -> None:
    learning.create_record(
        dataset_id,
        "answer",
        {
            "record_code": record,
            "student_code": student,
            "material_code": material,
            "theme_code": "T01",
            "node_code": "N01",
            "answer_event_code": event,
            "revision": revision,
            "submitted_at": "2026-01-20T09:00:00",
            "grading_status": status,
            "score": score,
            "full_score": 1,
        },
    )


def test_student_theme_metric_uses_latest_valid_answers_and_marks_limited_evidence(
    tmp_path: Path,
) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed_base(learning, dataset_id)
    _answer(
        learning,
        dataset_id,
        record="DR001",
        event="E01",
        material="Q001",
        score=0,
    )
    _answer(
        learning,
        dataset_id,
        record="DR002",
        event="E01",
        material="Q001",
        revision=2,
        score=1,
    )
    _answer(
        learning,
        dataset_id,
        record="DR003",
        event="E02",
        material="Q002",
        score=1,
    )
    _answer(
        learning,
        dataset_id,
        record="DR004",
        event="E03",
        material="Q003",
        status="待评分",
        score=None,
    )
    _answer(
        learning,
        dataset_id,
        record="DR005",
        event="E04",
        material="Q004",
        score=2,
    )
    _answer(
        learning,
        dataset_id,
        record="DR006",
        event="E05",
        material="Q003",
        status="未作答",
        score=None,
    )

    analytics = import_module("tailorlearn.analytics").AnalyticsService(repository)
    metric = analytics.student_theme_metrics(dataset_id, "D01", "N01")[0]

    assert metric["theme_code"] == "T01"
    assert metric["score_rate"] == 1.0
    assert metric["scored_records"] == 2
    assert metric["pending_records"] == 1
    assert metric["unanswered_records"] == 1
    assert metric["excluded_records"] == 1
    assert metric["evidence_status"] == "limited"


def test_theme_trend_requires_sufficient_evidence_at_both_nodes(tmp_path: Path) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed_base(learning, dataset_id)
    learning.create_record(
        dataset_id,
        "node",
        {
            "node_code": "N02",
            "test_date": "2026-06-17",
            "taught_topics": "T01",
            "description": "第二次测验",
            "available_materials": "Q001、Q002、Q003",
        },
    )
    for index, material in enumerate(("Q001", "Q002", "Q003"), start=1):
        _answer(
            learning,
            dataset_id,
            record=f"DR1{index:02d}",
            event=f"N01E{index}",
            material=material,
            score=1,
        )
        learning.create_record(
            dataset_id,
            "answer",
            {
                "record_code": f"DR2{index:02d}",
                "student_code": "D01",
                "material_code": material,
                "theme_code": "T01",
                "node_code": "N02",
                "answer_event_code": f"N02E{index}",
                "revision": 1,
                "submitted_at": "2026-06-17T09:00:00",
                "grading_status": "已评分",
                "score": 0,
                "full_score": 1,
            },
        )

    analytics = import_module("tailorlearn.analytics").AnalyticsService(repository)
    trend = analytics.student_theme_trend(dataset_id, "D01", "T01")

    assert trend["status"] == "observable"
    assert trend["delta"] == -1.0
    assert [(point["node_code"], point["score_rate"]) for point in trend["points"]] == [
        ("N01", 1.0),
        ("N02", 0.0),
    ]


def test_class_theme_metric_requires_five_students_and_ten_valid_records(
    tmp_path: Path,
) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed_base(learning, dataset_id)
    for student_index in range(2, 6):
        learning.create_record(
            dataset_id,
            "student",
            {
                "student_code": f"D{student_index:02d}",
                "class_code": "DC01",
                "grade": 7,
            },
        )
    for student_index in range(1, 6):
        for material_index in range(1, 3):
            learning.create_record(
                dataset_id,
                "answer",
                {
                    "record_code": f"DR{student_index}{material_index}0",
                    "student_code": f"D{student_index:02d}",
                    "material_code": f"Q{material_index:03d}",
                    "theme_code": "T01",
                    "node_code": "N01",
                    "answer_event_code": f"E{student_index}{material_index}",
                    "revision": 1,
                    "submitted_at": "2026-01-20T09:00:00",
                    "grading_status": "已评分",
                    "score": 1,
                    "full_score": 1,
                },
            )

    analytics = import_module("tailorlearn.analytics").AnalyticsService(repository)
    metric = analytics.class_theme_metrics(dataset_id, "N01")[0]

    assert metric["class_code"] == "DC01"
    assert metric["student_count"] == 5
    assert metric["scored_records"] == 10
    assert metric["score_rate"] == 1.0
    assert metric["evidence_status"] == "sufficient"


def test_supplemental_observation_uses_existing_unseen_material_for_missing_evidence(
    tmp_path: Path,
) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed_base(learning, dataset_id)
    _answer(
        learning,
        dataset_id,
        record="DR001",
        event="E01",
        material="Q001",
        score=1,
    )

    analytics = import_module("tailorlearn.analytics").AnalyticsService(repository)
    suggestions = analytics.supplemental_observations(dataset_id, "D01", "N01")

    assert suggestions == [
        {
            "theme_code": "T01",
            "indicator_code": "T01I1",
            "reason": "有效记录不足 3 条，当前仅有 1 条。",
            "candidate_material_codes": ["Q002", "Q003", "Q004"],
        }
    ]


def test_analysis_snapshot_reports_changed_metric_and_new_evidence(tmp_path: Path) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed_base(learning, dataset_id)
    _answer(
        learning,
        dataset_id,
        record="DR001",
        event="E01",
        material="Q001",
        score=1,
    )
    analytics = import_module("tailorlearn.analytics").AnalyticsService(repository)
    snapshot = analytics.create_snapshot(
        dataset_id,
        {"student_code": "D01", "node_code": "N01"},
    )
    _answer(
        learning,
        dataset_id,
        record="DR002",
        event="E02",
        material="Q002",
        score=0,
    )
    _answer(
        learning,
        dataset_id,
        record="DR003",
        event="E03",
        material="Q003",
        score=0,
    )

    comparison = analytics.compare_snapshot(snapshot["id"])

    assert comparison["affected_record_codes"] == ["DR002", "DR003"]
    assert comparison["changes"] == [
        {
            "theme_code": "T01",
            "before_score_rate": 1.0,
            "after_score_rate": 1 / 3,
            "before_evidence_status": "limited",
            "after_evidence_status": "sufficient",
        }
    ]


def test_list_snapshots_returns_saved_scope_for_dataset(tmp_path: Path) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed_base(learning, dataset_id)
    analytics = import_module("tailorlearn.analytics").AnalyticsService(repository)
    snapshot = analytics.create_snapshot(
        dataset_id,
        {"student_code": "D01", "node_code": "N01"},
    )

    snapshots = analytics.list_snapshots(dataset_id)

    assert snapshots == [
        {
            "id": snapshot["id"],
            "scope": {"student_code": "D01", "node_code": "N01"},
        }
    ]


def test_theme_evidence_exposes_valid_record_requirement_and_material(tmp_path: Path) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed_base(learning, dataset_id)
    _answer(
        learning,
        dataset_id,
        record="DR001",
        event="E01",
        material="Q001",
        score=1,
    )

    analytics = import_module("tailorlearn.analytics").AnalyticsService(repository)
    evidence = analytics.theme_evidence(dataset_id, "D01", "N01", "T01")

    assert evidence["requirement_ids"] == ["T01I1"]
    assert evidence["records"] == [
        {
            "record_code": "DR001",
            "material_code": "Q001",
            "score": 1,
            "full_score": 1,
            "indicator_codes": ["T01I1"],
        }
    ]


def test_student_indicator_metric_uses_material_indicator_mapping(tmp_path: Path) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed_base(learning, dataset_id)
    for index, score in enumerate((1, 0, 1), start=1):
        _answer(
            learning,
            dataset_id,
            record=f"DR{index:03d}",
            event=f"E{index:02d}",
            material=f"Q{index:03d}",
            score=score,
        )

    analytics = import_module("tailorlearn.analytics").AnalyticsService(repository)
    metric = analytics.student_indicator_metrics(dataset_id, "D01", "N01")[0]

    assert metric["indicator_code"] == "T01I1"
    assert metric["score_rate"] == 2 / 3
    assert metric["scored_records"] == 3
    assert metric["evidence_status"] == "sufficient"


def test_class_distribution_only_includes_students_with_sufficient_evidence(
    tmp_path: Path,
) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed_base(learning, dataset_id)
    for student_index in range(2, 6):
        learning.create_record(
            dataset_id,
            "student",
            {
                "student_code": f"D{student_index:02d}",
                "class_code": "DC01",
                "grade": 7,
            },
        )
    for student_index in range(1, 6):
        for material_index in range(1, 4):
            _answer(
                learning,
                dataset_id,
                record=f"DR{student_index}{material_index}0",
                event=f"E{student_index}{material_index}",
                material=f"Q{material_index:03d}",
                score=student_index % 2,
                student=f"D{student_index:02d}",
            )

    analytics = import_module("tailorlearn.analytics").AnalyticsService(repository)
    distribution = analytics.class_theme_distribution(dataset_id, "N01")

    assert distribution == [
        {
            "class_code": "DC01",
            "theme_code": "T01",
            "theme_name": "代数式",
            "observed_students": 5,
            "student_rates": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    ]


def test_typical_case_bundle_keeps_evidence_status_and_traceable_records(
    tmp_path: Path,
) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed_base(learning, dataset_id)
    _answer(
        learning,
        dataset_id,
        record="DR001",
        event="E01",
        material="Q001",
        score=1,
    )

    analytics = import_module("tailorlearn.analytics").AnalyticsService(repository)
    bundle = analytics.typical_case_bundle(dataset_id, "D01", "N01")

    assert bundle["student_code"] == "D01"
    assert bundle["themes"][0]["evidence_status"] == "limited"
    assert bundle["evidence"]["T01"]["records"][0]["record_code"] == "DR001"
