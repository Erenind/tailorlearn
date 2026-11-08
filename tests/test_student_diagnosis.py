from __future__ import annotations

from importlib import import_module
from pathlib import Path

from tailorlearn.analytics import AnalyticsService
from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.models import ProviderProfile


def _services(tmp_path: Path) -> tuple[Repository, LearningDataService, str]:
    repository = Repository(tmp_path / "diagnosis.sqlite3")
    repository.initialize()
    learning = LearningDataService(repository, tmp_path)
    dataset = learning.create_dataset("诊断测试数据")
    return repository, learning, dataset["id"]


def _seed(learning: LearningDataService, dataset_id: str) -> None:
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
    for index in range(1, 4):
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
            "available_materials": "Q001、Q002、Q003",
        },
    )
    learning.create_record(
        dataset_id,
        "answer",
        {
            "record_code": "DR001",
            "student_code": "D01",
            "material_code": "Q001",
            "theme_code": "T01",
            "node_code": "N01",
            "answer_event_code": "E01",
            "revision": 1,
            "submitted_at": "2026-01-20T09:00:00",
            "grading_status": "已评分",
            "score": 0,
            "full_score": 1,
            "final_answer": "不应发送给模型的原始答案",
        },
    )
    learning.create_record(
        dataset_id,
        "answer",
        {
            "record_code": "DR002",
            "student_code": "D01",
            "material_code": "Q002",
            "theme_code": "T01",
            "node_code": "N01",
            "answer_event_code": "E02",
            "revision": 1,
            "submitted_at": "2026-01-20T09:00:00",
            "grading_status": "待评分",
            "score": None,
            "full_score": 1,
        },
    )


def test_evidence_pack_excludes_raw_answers_and_keeps_only_traceable_candidates(
    tmp_path: Path,
) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed(learning, dataset_id)
    diagnosis = import_module("tailorlearn.student_diagnosis").StudentDiagnosisService(
        repository,
        AnalyticsService(repository),
    )

    pack = diagnosis.build_evidence_pack(dataset_id, "D01", "N01")

    assert pack["student_code"] == "D01"
    assert pack["node_code"] == "N01"
    assert pack["themes"] == [
        {
            "theme_code": "T01",
            "theme_name": "代数式",
            "score_rate": 0.0,
            "scored_records": 1,
            "pending_records": 1,
            "unanswered_records": 0,
            "evidence_status": "limited",
        }
    ]
    assert pack["evidence"] == {
        "T01": {
            "requirement_ids": ["T01I1"],
            "records": [
                {
                    "record_code": "DR001",
                    "material_code": "Q001",
                    "score": 0,
                    "full_score": 1,
                    "indicator_codes": ["T01I1"],
                }
            ],
        }
    }
    assert pack["candidate_material_codes"] == ["Q002", "Q003"]
    assert "不应发送给模型的原始答案" not in str(pack)


class _DiagnosisClient:
    def draft_student_diagnosis(self, evidence_pack: dict[str, object]) -> tuple[dict[str, object], str]:
        return (
            {
                "summary": "当前仅有一条有效记录，需补充观察后再作判断。",
                "observations": [
                    {
                        "theme_code": "T01",
                        "statement": "已观察到该题得分为 0/1。",
                        "evidence_record_codes": ["DR001"],
                        "indicator_codes": ["T01I1"],
                    }
                ],
                "information_gaps": [
                    {
                        "theme_code": "T01",
                        "statement": "有效记录不足三条。",
                    }
                ],
                "suggestions": [
                    {"material_code": "Q002", "reason": "补充同一观察指标的证据。"}
                ],
                "teacher_actions": ["观察补充作答后再更新诊断。"],
            },
            "{\"mock\": \"response\"}",
        )


class _FlakyDiagnosisClient(_DiagnosisClient):
    def __init__(self) -> None:
        self.calls = 0

    def draft_student_diagnosis(self, evidence_pack: dict[str, object]) -> tuple[dict[str, object], str]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary API failure")
        return super().draft_student_diagnosis(evidence_pack)


def test_batch_job_persists_a_reviewable_diagnosis_for_each_student(
    tmp_path: Path,
) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed(learning, dataset_id)
    diagnosis = import_module("tailorlearn.student_diagnosis").StudentDiagnosisService(
        repository,
        AnalyticsService(repository),
    )
    profile = ProviderProfile(
        id="provider_test",
        vendor_name="测试厂商",
        base_url="https://example.test/v1",
        model="test-model",
        api_key="secret",
    )

    job = diagnosis.create_batch_job(dataset_id, "DC01", "N01", profile)
    result = diagnosis.process_job(job["id"], _DiagnosisClient())
    drafts = diagnosis.list_diagnoses(dataset_id, "D01", "N01")

    assert result["status"] == "completed"
    assert result["completed"] == 1
    assert result["failed"] == 0
    assert drafts == [
        {
            "student_code": "D01",
            "node_code": "N01",
            "status": "needs_review",
            "summary": "当前仅有一条有效记录，需补充观察后再作判断。",
            "candidate_material_codes": ["Q002", "Q003"],
            "reviewed_summary": "",
        }
    ]


def test_single_student_job_does_not_enqueue_classmates(tmp_path: Path) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed(learning, dataset_id)
    learning.create_record(
        dataset_id,
        "student",
        {"student_code": "D02", "class_code": "DC01", "grade": 7},
    )
    diagnosis = import_module("tailorlearn.student_diagnosis").StudentDiagnosisService(
        repository,
        AnalyticsService(repository),
    )
    profile = ProviderProfile(
        id="provider_test",
        vendor_name="测试厂商",
        base_url="https://example.test/v1",
        model="test-model",
        api_key="secret",
    )

    job = diagnosis.create_student_job(dataset_id, "D01", "N01", profile)
    result = diagnosis.process_job(job["id"], _DiagnosisClient())

    assert result["total"] == 1
    assert result["completed"] == 1
    assert diagnosis.list_diagnoses(dataset_id, "D02", "N01") == []


def test_teacher_can_accept_a_diagnosis_with_a_revised_summary(tmp_path: Path) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed(learning, dataset_id)
    diagnosis = import_module("tailorlearn.student_diagnosis").StudentDiagnosisService(
        repository,
        AnalyticsService(repository),
    )
    profile = ProviderProfile(
        id="provider_test",
        vendor_name="测试厂商",
        base_url="https://example.test/v1",
        model="test-model",
        api_key="secret",
    )
    job = diagnosis.create_batch_job(dataset_id, "DC01", "N01", profile)
    diagnosis.process_job(job["id"], _DiagnosisClient())
    latest = diagnosis.get_latest_diagnosis(dataset_id, "D01", "N01")

    reviewed = diagnosis.review_diagnosis(
        latest["id"],
        status="accepted",
        reviewed_summary="教师确认：先补充 Q002 后再判断。",
        review_note="课堂观察后复核。",
    )

    assert reviewed["status"] == "accepted"
    assert reviewed["reviewed_summary"] == "教师确认：先补充 Q002 后再判断。"
    assert reviewed["review_note"] == "课堂观察后复核。"


def test_data_change_marks_existing_diagnosis_as_needing_update(tmp_path: Path) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed(learning, dataset_id)
    diagnosis = import_module("tailorlearn.student_diagnosis").StudentDiagnosisService(
        repository,
        AnalyticsService(repository),
    )
    profile = ProviderProfile(
        id="provider_test",
        vendor_name="测试厂商",
        base_url="https://example.test/v1",
        model="test-model",
        api_key="secret",
    )
    job = diagnosis.create_batch_job(dataset_id, "DC01", "N01", profile)
    diagnosis.process_job(job["id"], _DiagnosisClient())
    learning.create_record(
        dataset_id,
        "answer",
        {
            "record_code": "DR003",
            "student_code": "D01",
            "material_code": "Q003",
            "theme_code": "T01",
            "node_code": "N01",
            "answer_event_code": "E03",
            "revision": 1,
            "submitted_at": "2026-01-21T09:00:00",
            "grading_status": "已评分",
            "score": 1,
            "full_score": 1,
        },
    )

    changed = diagnosis.refresh_stale_diagnoses(dataset_id)
    latest = diagnosis.get_latest_diagnosis(dataset_id, "D01", "N01")

    assert changed == 1
    assert latest["status"] == "stale"


def test_failed_batch_item_can_be_queued_and_processed_again(tmp_path: Path) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed(learning, dataset_id)
    diagnosis = import_module("tailorlearn.student_diagnosis").StudentDiagnosisService(
        repository,
        AnalyticsService(repository),
    )
    profile = ProviderProfile(
        id="provider_test",
        vendor_name="测试厂商",
        base_url="https://example.test/v1",
        model="test-model",
        api_key="secret",
    )
    client = _FlakyDiagnosisClient()
    job = diagnosis.create_student_job(dataset_id, "D01", "N01", profile)

    failed = diagnosis.process_job(job["id"], client)
    requeued = diagnosis.retry_failed_items(job["id"])
    completed = diagnosis.process_job(job["id"], client)

    assert failed["status"] == "completed_with_errors"
    assert requeued == 1
    assert completed["status"] == "completed"
    assert completed["completed"] == 1


def test_class_job_list_exposes_progress_for_resuming(tmp_path: Path) -> None:
    repository, learning, dataset_id = _services(tmp_path)
    _seed(learning, dataset_id)
    diagnosis = import_module("tailorlearn.student_diagnosis").StudentDiagnosisService(
        repository,
        AnalyticsService(repository),
    )
    profile = ProviderProfile(
        id="provider_test",
        vendor_name="测试厂商",
        base_url="https://example.test/v1",
        model="test-model",
        api_key="secret",
    )
    job = diagnosis.create_batch_job(dataset_id, "DC01", "N01", profile)

    jobs = diagnosis.list_jobs(dataset_id, "DC01", "N01")

    assert jobs == [
        {
            "id": job["id"],
            "status": "queued",
            "total": 1,
            "completed": 0,
            "failed": 0,
        }
    ]
