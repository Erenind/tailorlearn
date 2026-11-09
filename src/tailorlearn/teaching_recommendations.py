from __future__ import annotations

from collections import defaultdict
from typing import Any

from tailorlearn.analytics import AnalyticsService
from tailorlearn.db import Repository
from tailorlearn.learning_data import split_multi_value


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
            "recommendation": None,
        }

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
