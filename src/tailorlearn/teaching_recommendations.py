from __future__ import annotations

from collections import defaultdict
import json
from typing import Any, Protocol
import uuid

from tailorlearn.analytics import AnalyticsService
from tailorlearn.db import Repository, utc_now
from tailorlearn.learning_data import split_multi_value
from tailorlearn.models import ProviderProfile, TeachingRecommendationDraft


class TeachingRecommendationClient(Protocol):
    def draft_teaching_recommendation(
        self,
        evidence_pack: dict[str, Any],
    ) -> tuple[dict[str, Any], str]: ...


class TeachingRecommendationService:
    """Build traceable class evidence before any teaching recommendation exists."""

    def __init__(self, repository: Repository, analytics: AnalyticsService):
        self.repo = repository
        self.analytics = analytics

    def build_evidence_pack(
        self,
        dataset_id: str,
        class_code: str,
        node_code: str,
    ) -> dict[str, Any]:
        student_rows = self.analytics._dataset_rows("learning_students", dataset_id)
        students = sorted(
            str(row["student_code"])
            for row in student_rows
            if str(row.get("class_code") or "未分班") == class_code
        )
        if not students:
            raise ValueError("该班级没有可分析的学生")
        class_themes = [
            item
            for item in self.analytics.class_theme_metrics(dataset_id, node_code)
            if item["class_code"] == class_code
        ]
        requirements = self.analytics._dataset_rows("learning_requirements", dataset_id)
        materials = self.analytics._dataset_rows("learning_materials", dataset_id)
        material_by_code = {str(item["material_code"]): item for item in materials}
        node = next(
            (
                item
                for item in self.analytics._dataset_rows("learning_nodes", dataset_id)
                if item["node_code"] == node_code
            ),
            None,
        )
        if node is None:
            raise ValueError("找不到指定测验节点")
        available_material_codes = set(split_multi_value(node.get("available_materials")))
        all_students = {str(row["student_code"]) for row in student_rows}
        all_nodes = {
            str(row["node_code"])
            for row in self.analytics._dataset_rows("learning_nodes", dataset_id)
        }
        themes = {
            str(row["theme_code"])
            for row in requirements
            if row.get("theme_code")
        }
        class_answers = [
            row
            for row in self.analytics._latest_answers(dataset_id)
            if row.get("node_code") == node_code
            and row.get("student_code") in students
        ]
        valid_answers = [
            row
            for row in class_answers
            if self.analytics._is_valid_scored_answer(
                row,
                students=all_students,
                materials=set(material_by_code),
                nodes=all_nodes,
                themes=themes,
            )
        ]
        blueprint = self._assessment_blueprint(
            requirements,
            material_by_code,
            available_material_codes,
            class_answers,
            valid_answers,
        )
        groups = self._groups(dataset_id, students, node_code, class_themes)
        candidates = self._candidate_materials(
            students,
            material_by_code,
            available_material_codes,
            valid_answers,
            requirements,
        )
        return {
            "dataset_id": dataset_id,
            "class_code": class_code,
            "node_code": node_code,
            "class_themes": class_themes,
            "assessment_blueprint": blueprint,
            "groups": groups,
            "candidate_materials": candidates,
        }

    def export_case_bundle(
        self,
        dataset_id: str,
        class_code: str,
        node_code: str,
    ) -> dict[str, Any]:
        """Return a de-identified export payload before rendering any format."""
        return {
            "dataset_id": dataset_id,
            "class_code": class_code,
            "node_code": node_code,
            "evidence_pack": self.build_evidence_pack(
                dataset_id, class_code, node_code
            ),
            "recommendation": self.get_latest_recommendation(
                dataset_id, class_code, node_code
            ),
        }

    def _change_log_cutoff(self, dataset_id: str) -> int:
        with self.repo.connect() as connection:
            return int(
                connection.execute(
                    "SELECT COALESCE(MAX(id), 0) FROM learning_change_log WHERE dataset_id=?",
                    (dataset_id,),
                ).fetchone()[0]
            )

    @staticmethod
    def _validate_draft(
        draft_data: dict[str, Any],
        evidence_pack: dict[str, Any],
    ) -> TeachingRecommendationDraft:
        draft = TeachingRecommendationDraft.model_validate(draft_data)
        forbidden_claims = ("未掌握", "掌握不足", "能力下降")
        if any(marker in json.dumps(draft.model_dump(), ensure_ascii=False) for marker in forbidden_claims):
            raise ValueError("信息不足时不得作出未掌握或能力变化判定")
        theme_codes = {
            str(item["theme_code"]) for item in evidence_pack["class_themes"]
        }
        indicator_codes = {
            str(item["indicator_code"])
            for item in evidence_pack["assessment_blueprint"]
        }
        groups = {
            (str(item["theme_code"]), str(item["group"])): set(item["student_codes"])
            for item in evidence_pack["groups"]
        }
        candidates = {
            (
                str(item["theme_code"]),
                str(item["indicator_code"]),
                str(item["material_code"]),
            ): set(item["eligible_student_codes"])
            for item in evidence_pack["candidate_materials"]
        }
        for observation in draft.observations:
            if observation.theme_code not in theme_codes:
                raise ValueError("AI 引用了范围外主题")
            if not set(observation.indicator_codes) <= indicator_codes:
                raise ValueError("AI 引用了范围外观察指标")
        for gap in draft.information_gaps:
            if gap.theme_code not in theme_codes or gap.indicator_code not in indicator_codes:
                raise ValueError("AI 引用了范围外数据缺口")
        for activity in draft.group_activities:
            allowed_students = groups.get((activity.theme_code, activity.group))
            if allowed_students is None or not set(activity.student_codes) <= allowed_students:
                raise ValueError("AI 引用了范围外分层学生")
        for suggestion in draft.material_suggestions:
            key = (
                suggestion.theme_code,
                suggestion.indicator_code,
                suggestion.material_code,
            )
            eligible_students = candidates.get(key)
            if (
                eligible_students is None
                or not suggestion.student_codes
                or not set(suggestion.student_codes) <= eligible_students
            ):
                raise ValueError("AI 推荐了范围外或已作答材料")
        return draft

    def create_recommendation(
        self,
        dataset_id: str,
        class_code: str,
        node_code: str,
        profile: ProviderProfile,
        client: TeachingRecommendationClient,
    ) -> dict[str, Any]:
        evidence_pack = self.build_evidence_pack(dataset_id, class_code, node_code)
        draft_data, raw_response = client.draft_teaching_recommendation(evidence_pack)
        draft = self._validate_draft(draft_data, evidence_pack)
        recommendation_id = f"atr_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        with self.repo.connect() as connection:
            connection.execute(
                """
                INSERT INTO ai_teaching_recommendations
                (id, dataset_id, class_code, node_code, change_log_cutoff,
                 evidence_pack_json, response_json, raw_response, provider_id,
                 provider_vendor, provider_base_url, provider_model, prompt_version,
                 status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'needs_review', ?, ?)
                """,
                (
                    recommendation_id,
                    dataset_id,
                    class_code,
                    node_code,
                    self._change_log_cutoff(dataset_id),
                    json.dumps(evidence_pack, ensure_ascii=False),
                    json.dumps(draft.model_dump(), ensure_ascii=False),
                    raw_response,
                    profile.id,
                    profile.vendor_name,
                    profile.base_url,
                    profile.model,
                    "class-teaching-v1",
                    now,
                    now,
                ),
            )
        result = self.get_latest_recommendation(dataset_id, class_code, node_code)
        if result is None:
            raise RuntimeError("教学建议草稿保存失败")
        return result

    def get_latest_recommendation(
        self,
        dataset_id: str,
        class_code: str,
        node_code: str,
    ) -> dict[str, Any] | None:
        with self.repo.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM ai_teaching_recommendations
                WHERE dataset_id=? AND class_code=? AND node_code=?
                ORDER BY created_at DESC, id DESC LIMIT 1
                """,
                (dataset_id, class_code, node_code),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "dataset_id": row["dataset_id"],
            "class_code": row["class_code"],
            "node_code": row["node_code"],
            "status": row["status"],
            "draft": json.loads(row["response_json"]),
            "evidence_pack": json.loads(row["evidence_pack_json"]),
            "reviewed_summary": row["reviewed_summary"],
            "review_note": row["review_note"],
        }

    def review_recommendation(
        self,
        recommendation_id: str,
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
                UPDATE ai_teaching_recommendations
                SET status=?, reviewed_summary=?, review_note=?, reviewed_at=?, updated_at=?
                WHERE id=?
                """,
                (
                    status,
                    reviewed_summary.strip(),
                    review_note.strip(),
                    now,
                    now,
                    recommendation_id,
                ),
            ).rowcount
            if not updated:
                raise ValueError("找不到教学建议草稿")
        return {"id": recommendation_id, "status": status, "reviewed_summary": reviewed_summary.strip(), "review_note": review_note.strip()}

    def refresh_stale_recommendations(self, dataset_id: str) -> int:
        cutoff = self._change_log_cutoff(dataset_id)
        with self.repo.connect() as connection:
            return connection.execute(
                """
                UPDATE ai_teaching_recommendations
                SET status='stale', updated_at=?
                WHERE dataset_id=? AND change_log_cutoff<?
                  AND status IN ('needs_review', 'accepted')
                """,
                (utc_now(), dataset_id, cutoff),
            ).rowcount

    @staticmethod
    def _assessment_blueprint(
        requirements: list[dict[str, Any]],
        material_by_code: dict[str, dict[str, Any]],
        available_codes: set[str],
        answers: list[dict[str, Any]],
        valid_answers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for requirement in sorted(requirements, key=lambda item: str(item["indicator_code"])):
            indicator_code = str(requirement["indicator_code"])
            material_codes = [
                code
                for code, material in sorted(material_by_code.items())
                if code in available_codes
                and indicator_code in split_multi_value(material.get("indicator_codes"))
            ]
            matching = [
                answer for answer in answers if answer.get("material_code") in material_codes
            ]
            valid_count = sum(answer in valid_answers for answer in matching)
            pending_count = sum(answer.get("grading_status") == "待评分" for answer in matching)
            unanswered_count = sum(answer.get("grading_status") == "未作答" for answer in matching)
            coverage = (
                "observed"
                if valid_count
                else "pending_review"
                if pending_count
                else "unanswered_only"
                if unanswered_count
                else "no_available_material"
                if not material_codes
                else "needs_observation"
            )
            items.append(
                {
                    "theme_code": requirement.get("theme_code"),
                    "indicator_code": indicator_code,
                    "available_material_codes": material_codes,
                    "valid_answer_count": valid_count,
                    "pending_count": pending_count,
                    "unanswered_count": unanswered_count,
                    "coverage_status": coverage,
                }
            )
        return items

    def _groups(
        self,
        dataset_id: str,
        students: list[str],
        node_code: str,
        class_themes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for class_theme in class_themes:
            theme_code = str(class_theme["theme_code"])
            class_rate = class_theme["score_rate"]
            for student_code in students:
                metric = next(
                    (
                        item
                        for item in self.analytics.student_theme_metrics(
                            dataset_id, student_code, node_code
                        )
                        if item["theme_code"] == theme_code
                    ),
                    None,
                )
                if (
                    class_theme["evidence_status"] != "sufficient"
                    or metric is None
                    or metric["evidence_status"] != "sufficient"
                    or metric["score_rate"] is None
                ):
                    group = "needs_observation"
                elif metric["score_rate"] >= class_rate:
                    group = "observed_higher"
                elif metric["score_rate"] <= class_rate - 0.20:
                    group = "needs_attention"
                else:
                    group = "observed_other"
                grouped[(theme_code, group)].append(student_code)
        reasons = {
            "observed_higher": "学生主题证据充分，已观察得分率不低于班级同主题得分率。",
            "needs_attention": "学生主题证据充分，已观察得分率至少低于班级同主题得分率 20 个百分点。",
            "observed_other": "学生主题证据充分，已观察得分率接近班级同主题得分率。",
            "needs_observation": "学生或班级主题证据不足，需补充观察后再分层。",
        }
        return [
            {
                "theme_code": theme_code,
                "group": group,
                "student_codes": codes,
                "reason": reasons[group],
            }
            for (theme_code, group), codes in sorted(grouped.items())
        ]

    @staticmethod
    def _candidate_materials(
        students: list[str],
        material_by_code: dict[str, dict[str, Any]],
        available_codes: set[str],
        valid_answers: list[dict[str, Any]],
        requirements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        answered = {
            (str(answer["student_code"]), str(answer["material_code"]))
            for answer in valid_answers
        }
        candidates: list[dict[str, Any]] = []
        for requirement in sorted(requirements, key=lambda item: str(item["indicator_code"])):
            indicator_code = str(requirement["indicator_code"])
            for material_code, material in sorted(material_by_code.items()):
                if (
                    material_code not in available_codes
                    or material.get("material_type") not in {"题目", "学习卡"}
                    or indicator_code not in split_multi_value(material.get("indicator_codes"))
                ):
                    continue
                eligible = [
                    student_code
                    for student_code in students
                    if (student_code, material_code) not in answered
                ]
                if eligible:
                    candidates.append(
                        {
                            "theme_code": requirement.get("theme_code"),
                            "indicator_code": indicator_code,
                            "material_code": material_code,
                            "eligible_student_codes": eligible,
                        }
                    )
        return candidates
