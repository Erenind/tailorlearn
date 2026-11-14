from __future__ import annotations

from typing import Any

from tailorlearn.analytics import AnalyticsService
from tailorlearn.learning_data import LearningDataService
from tailorlearn.teacher_workbench import TeacherWorkbenchService


class HomeDashboardService:
    """Build a compact, evidence-safe overview for the navigation home."""

    def __init__(
        self,
        learning: LearningDataService,
        analytics: AnalyticsService,
        workbench: TeacherWorkbenchService,
    ):
        self.learning = learning
        self.analytics = analytics
        self.workbench = workbench

    def build(self, dataset_id: str) -> dict[str, Any]:
        board = self.workbench.build_todos(dataset_id)
        themes = self.analytics.class_theme_metrics(dataset_id)
        sufficient = sum(
            item["evidence_status"] == "sufficient" for item in themes
        )
        follow_up = (
            int(board["summary"]["supplemental_observation"])
            + int(board["summary"]["ai_review"])
            + int(board["summary"]["needs_update"])
        )
        return {
            "dataset_id": dataset_id,
            "metrics": {
                "student_count": len(
                    self.learning.list_records(dataset_id, "student")
                ),
                "valid_scored_answers": sum(
                    int(item["scored_records"]) for item in themes
                ),
                "data_processing_todos": int(
                    board["summary"]["data_processing"]
                ),
                "follow_up_todos": follow_up,
            },
            "evidence_snapshot": [
                {"label": "证据充分", "count": sufficient},
                {"label": "待补充观察", "count": len(themes) - sufficient},
            ],
            "todos": list(board["todos"]),
        }
