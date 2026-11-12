from __future__ import annotations

import json
import uuid
from typing import Any, Protocol

from tailorlearn.analytics import AnalyticsService
from tailorlearn.db import Repository, utc_now
from tailorlearn.models import ProviderProfile, StudentDiagnosisDraft


class DiagnosisClient(Protocol):
    def draft_student_diagnosis(
        self,
        evidence_pack: dict[str, Any],
    ) -> tuple[dict[str, Any], str]: ...


class StudentDiagnosisService:
    """Prepare the minimum, traceable evidence an AI diagnosis may receive."""

    def __init__(self, repository: Repository, analytics: AnalyticsService):
        self.repo = repository
        self.analytics = analytics

    def build_evidence_pack(
        self,
        dataset_id: str,
        student_code: str,
        node_code: str,
    ) -> dict[str, Any]:
        metrics = self.analytics.student_theme_metrics(
            dataset_id, student_code, node_code
        )
        themes = [
            {
                key: metric[key]
                for key in (
                    "theme_code",
                    "theme_name",
                    "score_rate",
                    "scored_records",
                    "pending_records",
                    "unanswered_records",
                    "evidence_status",
                )
            }
            for metric in metrics
        ]
        evidence = {
            str(metric["theme_code"]): self.analytics.theme_evidence(
                dataset_id,
                student_code,
                node_code,
                str(metric["theme_code"]),
            )
            for metric in metrics
        }
        candidate_material_codes: list[str] = []
        for suggestion in self.analytics.supplemental_observations(
            dataset_id, student_code, node_code
        ):
            for code in suggestion["candidate_material_codes"]:
                if code not in candidate_material_codes:
                    candidate_material_codes.append(code)
        return {
            "dataset_id": dataset_id,
            "student_code": student_code,
            "node_code": node_code,
            "themes": themes,
            "indicators": self.analytics.student_indicator_metrics(
                dataset_id, student_code, node_code
            ),
            "evidence": evidence,
            "candidate_material_codes": candidate_material_codes,
        }

    def _change_log_cutoff(self, dataset_id: str) -> int:
        with self.repo.connect() as connection:
            return int(
                connection.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM learning_change_log WHERE dataset_id=?",
                    (dataset_id,),
                ).fetchone()[0]
            )

    def create_batch_job(
        self,
        dataset_id: str,
        class_code: str,
        node_code: str,
        profile: ProviderProfile,
    ) -> dict[str, Any]:
        with self.repo.connect() as connection:
            students = [
                str(row["student_code"])
                for row in connection.execute(
                    """
                    SELECT student_code FROM learning_students
                    WHERE dataset_id=? AND class_code=? AND deleted_at IS NULL
                    ORDER BY student_code
                    """,
                    (dataset_id, class_code),
                ).fetchall()
            ]
            node_exists = connection.execute(
                """
                SELECT 1 FROM learning_nodes
                WHERE dataset_id=? AND node_code=? AND deleted_at IS NULL
                """,
                (dataset_id, node_code),
            ).fetchone()
            if not node_exists:
                raise ValueError("找不到指定测验节点")
            if not students:
                raise ValueError("该班级没有可诊断的学生")
            now = utc_now()
            job_id = f"adj_{uuid.uuid4().hex[:12]}"
            cutoff = self._change_log_cutoff(dataset_id)
            connection.execute(
                """
                INSERT INTO ai_diagnosis_jobs
                (id, dataset_id, class_code, node_code, change_log_cutoff,
                 provider_id, provider_vendor, provider_base_url, provider_model,
                 status, total, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    job_id,
                    dataset_id,
                    class_code,
                    node_code,
                    cutoff,
                    profile.id,
                    profile.vendor_name,
                    profile.base_url,
                    profile.model,
                    len(students),
                    now,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO ai_diagnosis_job_items
                (job_id, student_code, status, updated_at)
                VALUES (?, ?, 'queued', ?)
                """,
                [(job_id, student_code, now) for student_code in students],
            )
        return self.get_job(job_id)

    def create_student_job(
        self,
        dataset_id: str,
        student_code: str,
        node_code: str,
        profile: ProviderProfile,
    ) -> dict[str, Any]:
        with self.repo.connect() as connection:
            student = connection.execute(
                """
                SELECT class_code FROM learning_students
                WHERE dataset_id=? AND student_code=? AND deleted_at IS NULL
                """,
                (dataset_id, student_code),
            ).fetchone()
            node_exists = connection.execute(
                """
                SELECT 1 FROM learning_nodes
                WHERE dataset_id=? AND node_code=? AND deleted_at IS NULL
                """,
                (dataset_id, node_code),
            ).fetchone()
            if student is None:
                raise ValueError("找不到指定学生")
            if not node_exists:
                raise ValueError("找不到指定测验节点")
            now = utc_now()
            job_id = f"adj_{uuid.uuid4().hex[:12]}"
            cutoff = self._change_log_cutoff(dataset_id)
            connection.execute(
                """
                INSERT INTO ai_diagnosis_jobs
                (id, dataset_id, class_code, node_code, change_log_cutoff,
                 provider_id, provider_vendor, provider_base_url, provider_model,
                 status, total, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 1, ?, ?)
                """,
                (
                    job_id,
                    dataset_id,
                    str(student["class_code"] or "未分班"),
                    node_code,
                    cutoff,
                    profile.id,
                    profile.vendor_name,
                    profile.base_url,
                    profile.model,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO ai_diagnosis_job_items
                (job_id, student_code, status, updated_at)
                VALUES (?, ?, 'queued', ?)
                """,
                (job_id, student_code, now),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.repo.connect() as connection:
            row = connection.execute(
                "SELECT * FROM ai_diagnosis_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise ValueError("找不到诊断任务")
            return dict(row)

    def list_jobs(
        self,
        dataset_id: str,
        class_code: str,
        node_code: str,
    ) -> list[dict[str, Any]]:
        with self.repo.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, status, total, completed, failed
                FROM ai_diagnosis_jobs
                WHERE dataset_id=? AND class_code=? AND node_code=?
                ORDER BY created_at DESC, id DESC
                """,
                (dataset_id, class_code, node_code),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _validate_draft(
        draft_data: dict[str, Any],
        evidence_pack: dict[str, Any],
    ) -> StudentDiagnosisDraft:
        draft = StudentDiagnosisDraft.model_validate(draft_data)
        theme_codes = {str(item["theme_code"]) for item in evidence_pack["themes"]}
        record_codes = {
            str(record["record_code"])
            for evidence in evidence_pack["evidence"].values()
            for record in evidence["records"]
        }
        indicator_codes = {
            str(code)
            for evidence in evidence_pack["evidence"].values()
            for code in evidence["requirement_ids"]
        }
        candidate_codes = set(evidence_pack["candidate_material_codes"])
        weak_themes = {
            str(item["theme_code"])
            for item in evidence_pack["themes"]
            if item["evidence_status"] != "sufficient"
        }
        forbidden_claims = ("未掌握", "掌握不足", "能力下降")
        for observation in draft.observations:
            if observation.theme_code not in theme_codes:
                raise ValueError("AI 引用了范围外主题")
            if not set(observation.evidence_record_codes) <= record_codes:
                raise ValueError("AI 引用了不存在的作答记录")
            if not set(observation.indicator_codes) <= indicator_codes:
                raise ValueError("AI 引用了不存在的观察指标")
            if (
                observation.theme_code in weak_themes
                and any(marker in observation.statement for marker in forbidden_claims)
            ):
                raise ValueError("信息不足时不得作出未掌握或能力变化判定")
        for gap in draft.information_gaps:
            if gap.theme_code not in theme_codes:
                raise ValueError("AI 引用了范围外主题")
        for suggestion in draft.suggestions:
            if suggestion.material_code not in candidate_codes:
                raise ValueError("AI 推荐了范围外或已作答材料")
        return draft

    def process_job(
        self,
        job_id: str,
        client: DiagnosisClient,
    ) -> dict[str, Any]:
        job = self.get_job(job_id)
        now = utc_now()
        with self.repo.connect() as connection:
            connection.execute(
                "UPDATE ai_diagnosis_jobs SET status='running', updated_at=? WHERE id=?",
                (now, job_id),
            )
            items = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT * FROM ai_diagnosis_job_items
                    WHERE job_id=? AND status='queued'
                    ORDER BY id
                    """,
                    (job_id,),
                ).fetchall()
            ]
        for item in items:
            item_now = utc_now()
            with self.repo.connect() as connection:
                connection.execute(
                    """
                    UPDATE ai_diagnosis_job_items
                    SET status='running', attempts=attempts+1, error=NULL, updated_at=?
                    WHERE id=?
                    """,
                    (item_now, item["id"]),
                )
            try:
                evidence_pack = self.build_evidence_pack(
                    str(job["dataset_id"]),
                    str(item["student_code"]),
                    str(job["node_code"]),
                )
                draft_data, raw_response = client.draft_student_diagnosis(evidence_pack)
                draft = self._validate_draft(draft_data, evidence_pack)
                diagnosis_id = f"asd_{uuid.uuid4().hex[:12]}"
                saved_at = utc_now()
                with self.repo.connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO ai_student_diagnoses
                        (id, job_id, dataset_id, student_code, node_code,
                         change_log_cutoff, evidence_pack_json, response_json,
                         raw_response, provider_id, provider_vendor,
                         provider_base_url, provider_model, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                'needs_review', ?, ?)
                        """,
                        (
                            diagnosis_id,
                            job_id,
                            job["dataset_id"],
                            item["student_code"],
                            job["node_code"],
                            job["change_log_cutoff"],
                            json.dumps(evidence_pack, ensure_ascii=False),
                            json.dumps(draft.model_dump(), ensure_ascii=False),
                            raw_response,
                            job["provider_id"],
                            job["provider_vendor"],
                            job["provider_base_url"],
                            job["provider_model"],
                            saved_at,
                            saved_at,
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE ai_diagnosis_job_items
                        SET status='completed', diagnosis_id=?, updated_at=?
                        WHERE id=?
                        """,
                        (diagnosis_id, saved_at, item["id"]),
                    )
            except Exception as exc:
                with self.repo.connect() as connection:
                    connection.execute(
                        """
                        UPDATE ai_diagnosis_job_items
                        SET status='failed', error=?, updated_at=? WHERE id=?
                        """,
                        (str(exc), utc_now(), item["id"]),
                    )
        with self.repo.connect() as connection:
            summary = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
                    SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN status IN ('queued', 'running') THEN 1 ELSE 0 END) AS pending
                FROM ai_diagnosis_job_items WHERE job_id=?
                """,
                (job_id,),
            ).fetchone()
            completed = int(summary["completed"] or 0)
            failed = int(summary["failed"] or 0)
            pending = int(summary["pending"] or 0)
            status = (
                "running"
                if pending
                else "completed"
                if not failed
                else "completed_with_errors"
            )
            finished_at = utc_now() if not pending else None
            connection.execute(
                """
                UPDATE ai_diagnosis_jobs
                SET status=?, completed=?, failed=?, updated_at=?, finished_at=?
                WHERE id=?
                """,
                (status, completed, failed, utc_now(), finished_at, job_id),
            )
        return self.get_job(job_id)

    def list_diagnoses(
        self,
        dataset_id: str,
        student_code: str,
        node_code: str,
    ) -> list[dict[str, Any]]:
        with self.repo.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM ai_student_diagnoses
                WHERE dataset_id=? AND student_code=? AND node_code=?
                ORDER BY created_at DESC, id DESC
                """,
                (dataset_id, student_code, node_code),
            ).fetchall()
        diagnoses: list[dict[str, Any]] = []
        for row in rows:
            response = json.loads(row["response_json"])
            evidence_pack = json.loads(row["evidence_pack_json"])
            diagnoses.append(
                {
                    "student_code": row["student_code"],
                    "node_code": row["node_code"],
                    "status": row["status"],
                    "summary": response["summary"],
                    "candidate_material_codes": evidence_pack[
                        "candidate_material_codes"
                    ],
                    "reviewed_summary": row["reviewed_summary"],
                }
            )
        return diagnoses

    def get_latest_diagnosis(
        self,
        dataset_id: str,
        student_code: str,
        node_code: str,
    ) -> dict[str, Any] | None:
        with self.repo.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM ai_student_diagnoses
                WHERE dataset_id=? AND student_code=? AND node_code=?
                ORDER BY created_at DESC, id DESC LIMIT 1
                """,
                (dataset_id, student_code, node_code),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "student_code": row["student_code"],
            "node_code": row["node_code"],
            "status": row["status"],
            "draft": json.loads(row["response_json"]),
            "evidence_pack": json.loads(row["evidence_pack_json"]),
            "reviewed_summary": row["reviewed_summary"],
            "review_note": row["review_note"],
        }

    def review_diagnosis(
        self,
        diagnosis_id: str,
        *,
        status: str,
        reviewed_summary: str,
        review_note: str,
    ) -> dict[str, Any]:
        if status not in {"accepted", "rejected"}:
            raise ValueError("审核状态只能是 accepted 或 rejected")
        now = utc_now()
        with self.repo.connect() as connection:
            updated = connection.execute(
                """
                UPDATE ai_student_diagnoses
                SET status=?, reviewed_summary=?, review_note=?, reviewed_at=?, updated_at=?
                WHERE id=?
                """,
                (
                    status,
                    reviewed_summary.strip(),
                    review_note.strip(),
                    now,
                    now,
                    diagnosis_id,
                ),
            ).rowcount
            if not updated:
                raise ValueError("找不到诊断草稿")
        with self.repo.connect() as connection:
            row = connection.execute(
                "SELECT * FROM ai_student_diagnoses WHERE id=?", (diagnosis_id,)
            ).fetchone()
        return {
            "id": row["id"],
            "status": row["status"],
            "reviewed_summary": row["reviewed_summary"],
            "review_note": row["review_note"],
        }

    def refresh_stale_diagnoses(self, dataset_id: str) -> int:
        """Mark displayed drafts stale when later learning-data changes exist."""
        cutoff = self._change_log_cutoff(dataset_id)
        with self.repo.connect() as connection:
            return connection.execute(
                """
                UPDATE ai_student_diagnoses
                SET status='stale', updated_at=?
                WHERE dataset_id=? AND change_log_cutoff<?
                  AND status IN ('needs_review', 'accepted')
                """,
                (utc_now(), dataset_id, cutoff),
            ).rowcount

    def retry_failed_items(self, job_id: str) -> int:
        """Return failed students to the persisted queue without losing attempts."""
        now = utc_now()
        with self.repo.connect() as connection:
            retried = connection.execute(
                """
                UPDATE ai_diagnosis_job_items
                SET status='queued', error=NULL, updated_at=?
                WHERE job_id=? AND status='failed'
                """,
                (now, job_id),
            ).rowcount
            if retried:
                connection.execute(
                    """
                    UPDATE ai_diagnosis_jobs
                    SET status='queued', finished_at=NULL, updated_at=?
                    WHERE id=?
                    """,
                    (now, job_id),
                )
        return retried
