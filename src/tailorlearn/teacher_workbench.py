from __future__ import annotations

from typing import Any

from tailorlearn.analytics import AnalyticsService
from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.student_diagnosis import StudentDiagnosisService
from tailorlearn.teaching_recommendations import TeachingRecommendationService


WORKBENCH_CATEGORIES = (
    "data_processing",
    "supplemental_observation",
    "ai_review",
    "needs_update",
)


class TeacherWorkbenchService:
    """Aggregate existing teacher actions without creating new learner judgments."""

    def __init__(
        self,
        repository: Repository,
        learning: LearningDataService,
        analytics: AnalyticsService,
        diagnoses: StudentDiagnosisService,
        recommendations: TeachingRecommendationService,
    ):
        self.repo = repository
        self.learning = learning
        self.analytics = analytics
        self.diagnoses = diagnoses
        self.recommendations = recommendations

    @staticmethod
    def _todo(
        *,
        category: str,
        title: str,
        reason: str,
        target_page: str,
        context: dict[str, str | None],
        priority: int,
    ) -> dict[str, Any]:
        return {
            "category": category,
            "title": title,
            "reason": reason,
            "target_page": target_page,
            "context": context,
            "priority": priority,
        }

    def _latest_diagnosis_rows(
        self,
        dataset_id: str,
    ) -> list[dict[str, Any]]:
        with self.repo.connect() as connection:
            rows = connection.execute(
                """
                SELECT current.id, current.student_code, current.node_code,
                       current.status
                FROM ai_student_diagnoses AS current
                WHERE current.dataset_id=?
                  AND current.status IN ('needs_review', 'stale')
                  AND NOT EXISTS (
                      SELECT 1
                      FROM ai_student_diagnoses AS newer
                      WHERE newer.dataset_id=current.dataset_id
                        AND newer.student_code=current.student_code
                        AND newer.node_code=current.node_code
                        AND (
                            newer.created_at > current.created_at
                            OR (
                                newer.created_at=current.created_at
                                AND newer.id > current.id
                            )
                        )
                  )
                ORDER BY current.student_code, current.node_code
                """,
                (dataset_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _latest_recommendation_rows(
        self,
        dataset_id: str,
    ) -> list[dict[str, Any]]:
        with self.repo.connect() as connection:
            rows = connection.execute(
                """
                SELECT current.id, current.class_code, current.node_code,
                       current.status
                FROM ai_teaching_recommendations AS current
                WHERE current.dataset_id=?
                  AND current.status IN ('needs_review', 'stale')
                  AND NOT EXISTS (
                      SELECT 1
                      FROM ai_teaching_recommendations AS newer
                      WHERE newer.dataset_id=current.dataset_id
                        AND newer.class_code=current.class_code
                        AND newer.node_code=current.node_code
                        AND (
                            newer.created_at > current.created_at
                            OR (
                                newer.created_at=current.created_at
                                AND newer.id > current.id
                            )
                        )
                  )
                ORDER BY current.class_code, current.node_code
                """,
                (dataset_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def build_todos(self, dataset_id: str) -> dict[str, Any]:
        """Return only actionable metadata, never raw answers or mastery labels."""
        self.diagnoses.refresh_stale_diagnoses(dataset_id)
        self.recommendations.refresh_stale_recommendations(dataset_id)
        todos: list[dict[str, Any]] = []

        latest_answers = self.learning.list_records(
            dataset_id,
            "answer",
            latest_answers=True,
        )
        for answer in latest_answers:
            if answer.get("grading_status") != "待评分":
                continue
            todos.append(
                self._todo(
                    category="data_processing",
                    title=f"待评分作答 · {answer.get('record_code') or answer['row_id']}",
                    reason="仅统计最新修订；待评分记录不参与学习表现或掌握判定。",
                    target_page="材料与数据",
                    context={
                        "dataset_id": dataset_id,
                        "node_code": str(answer.get("node_code") or "") or None,
                        "student_code": str(answer.get("student_code") or "") or None,
                        "class_code": None,
                        "focus": "pending_answers",
                    },
                    priority=1,
                )
            )

        for issue in self.learning.quality_report(dataset_id)["issues"]:
            if issue.get("数据类型") != "作答记录":
                continue
            todos.append(
                self._todo(
                    category="data_processing",
                    title=f"作答数据异常 · {issue.get('记录') or issue.get('row_id')}",
                    reason=str(issue.get("说明") or "请核验该作答记录。"),
                    target_page="材料与数据",
                    context={
                        "dataset_id": dataset_id,
                        "node_code": None,
                        "student_code": None,
                        "class_code": None,
                        "focus": "answer_anomalies",
                    },
                    priority=1,
                )
            )

        for metric in self.analytics.class_theme_metrics(dataset_id):
            if metric.get("evidence_status") == "sufficient":
                continue
            todos.append(
                self._todo(
                    category="supplemental_observation",
                    title=(
                        f"补充观察 · {metric['class_code']} · "
                        f"{metric['theme_name']}"
                    ),
                    reason=(
                        f"当前为{metric['evidence_status']}证据：有效评分 "
                        f"{metric['scored_records']} 条，待评分 "
                        f"{metric['pending_records']} 条，未作答 "
                        f"{metric['unanswered_records']} 条。"
                        "信息不足仅提示补充观察，不作掌握结论。"
                    ),
                    target_page="班级分析",
                    context={
                        "dataset_id": dataset_id,
                        "node_code": None,
                        "student_code": None,
                        "class_code": str(metric["class_code"]),
                        "focus": "supplemental_observation",
                    },
                    priority=2,
                )
            )

        for row in self._latest_diagnosis_rows(dataset_id):
            category = "ai_review" if row["status"] == "needs_review" else "needs_update"
            status_reason = (
                "AI 草稿已生成，需由教师复核后使用。"
                if category == "ai_review"
                else "数据已更新，原 AI 草稿已失效，需重新核验或生成。"
            )
            todos.append(
                self._todo(
                    category=category,
                    title=f"学生诊断 · {row['student_code']} · {row['node_code']}",
                    reason=status_reason,
                    target_page="学生分析",
                    context={
                        "dataset_id": dataset_id,
                        "node_code": str(row["node_code"]),
                        "student_code": str(row["student_code"]),
                        "class_code": None,
                        "focus": "diagnosis_review",
                    },
                    priority=3 if category == "ai_review" else 4,
                )
            )

        for row in self._latest_recommendation_rows(dataset_id):
            category = "ai_review" if row["status"] == "needs_review" else "needs_update"
            status_reason = (
                "AI 教学建议草稿已生成，需由教师复核后使用。"
                if category == "ai_review"
                else "数据已更新，原 AI 教学建议已失效，需重新核验或生成。"
            )
            todos.append(
                self._todo(
                    category=category,
                    title=f"教学建议 · {row['class_code']} · {row['node_code']}",
                    reason=status_reason,
                    target_page="教学建议",
                    context={
                        "dataset_id": dataset_id,
                        "node_code": str(row["node_code"]),
                        "student_code": None,
                        "class_code": str(row["class_code"]),
                        "focus": "recommendation_review",
                    },
                    priority=3 if category == "ai_review" else 4,
                )
            )

        todos.sort(
            key=lambda item: (
                int(item["priority"]),
                str(item["title"]),
            )
        )
        summary = {
            category: sum(item["category"] == category for item in todos)
            for category in WORKBENCH_CATEGORIES
        }
        return {"dataset_id": dataset_id, "summary": summary, "todos": todos}
