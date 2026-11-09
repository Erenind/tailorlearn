from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tailorlearn.db import Repository


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
