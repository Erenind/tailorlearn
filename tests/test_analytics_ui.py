from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.paths import AppPaths


def test_streamlit_renders_student_and_class_analysis_navigation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    learning = LearningDataService(repository, paths.db_dir)
    dataset = learning.create_dataset("界面测试数据")
    dataset_id = dataset["id"]
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
            "theme_name": "主题一",
            "theme_goal": "目标",
            "indicator_code": "T01I1",
            "indicator_name": "指标一",
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
            "score": 1,
            "full_score": 1,
        },
    )
    monkeypatch.setattr(AppPaths, "default", classmethod(lambda cls: paths))
    st.cache_resource.clear()
    app_file = Path(__file__).resolve().parents[1] / "src" / "tailorlearn" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=20).run()

    assert "学生分析" in at.radio(key="main_navigation").options
    assert "班级分析" in at.radio(key="main_navigation").options
    assert "教学建议" in at.radio(key="main_navigation").options
    at.radio(key="main_navigation").set_value("学生分析").run()
    assert not at.exception
    assert any(item.value == "学生分析" for item in at.title)
    assert any(item.label == "AI 诊断与教师复核" for item in at.tabs)
    st.cache_resource.clear()
