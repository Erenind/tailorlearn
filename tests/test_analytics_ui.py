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
    learning.create_record(
        dataset_id,
        "student",
        {"student_code": "D06", "class_code": "DC01", "grade": 7},
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
    for material_index in range(1, 4):
        learning.create_record(
            dataset_id,
            "material",
            {
                "material_code": f"Q{material_index:03d}",
                "theme_code": "T01",
                "material_type": "题目",
                "question_type": "填空",
                "content": f"题目 {material_index}",
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
            "available_materials": "Q001、Q002、Q003",
        },
    )
    for student_index in range(1, 6):
        for material_index in range(1, 4):
            learning.create_record(
                dataset_id,
                "answer",
                {
                    "record_code": (
                        f"DR{student_index:02d}"
                        f"{material_index:02d}"
                    ),
                    "student_code": f"D{student_index:02d}",
                    "material_code": f"Q{material_index:03d}",
                    "theme_code": "T01",
                    "node_code": "N01",
                    "answer_event_code": (
                        f"E{student_index:02d}"
                        f"{material_index:02d}"
                    ),
                    "revision": 1,
                    "submitted_at": "2026-01-20T09:00:00",
                    "grading_status": "已评分",
                    "score": 1 if student_index == 1 else 0,
                    "full_score": 1,
                },
            )
    learning.create_record(
        dataset_id,
        "answer",
        {
            "record_code": "DR0601",
            "student_code": "D06",
            "material_code": "Q001",
            "theme_code": "T01",
            "node_code": "N01",
            "answer_event_code": "E0601",
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

    assert "学生分析" in at.button(key="nav_student").label
    assert "班级分析" in at.button(key="nav_class").label
    assert "教学建议" in at.button(key="nav_teaching").label
    at.button(key="nav_student").click().run()
    assert not at.exception
    assert any(item.value == "学生分析" for item in at.title)
    assert at.button(key="student_analysis_nav_overview").label.endswith("数据概览")
    assert at.button(key="student_analysis_nav_issues").label.endswith("问题分析")
    assert any(item.value == "数据统计标签" for item in at.subheader)
    ranking = at.dataframe(
        key=f"analytics_student_ranking_table_{dataset_id}"
    ).value
    assert ranking.loc[0, "综合名次"] == "1"
    assert ranking.loc[0, "学生编号"] == "D01"
    assert ranking.loc[0, "综合得分率"] == 1.0
    assert "D06" not in set(ranking["学生编号"])
    incomplete = at.dataframe(
        key=f"analytics_student_incomplete_ranking_table_{dataset_id}"
    ).value
    assert incomplete.loc[0, "学生编号"] == "D06"
    assert incomplete.loc[0, "记录完整度"] == "0/1（0%）"
    assert incomplete.loc[0, "已观察主题"] == "1/1"

    at.button(key="student_analysis_nav_issues").click().run()
    assert not at.exception
    assert any(item.value == "需要巩固的内容" for item in at.subheader)
    assert any(item.value == "个性化问题" for item in at.subheader)
    assert not any(item.value == "共性问题" for item in at.subheader)
    assert any(
        item.label == "AI 诊断与教师复核"
        for item in at.expander
    )

    at.button(key="nav_class").click().run()
    assert not at.exception
    assert at.button(key="class_analysis_nav_compare").label.endswith("班级对比")
    assert at.button(key="class_analysis_nav_specific").label.endswith(
        "具体班级分析"
    )
    assert at.get("plotly_chart")

    at.button(key="class_analysis_nav_specific").click().run()
    assert not at.exception
    assert at.selectbox(
        key=f"analytics_specific_class_{dataset_id}_all"
    ).value == "DC01"
    assert any(item.value == "共性问题" for item in at.subheader)
    assert any(item.value == "待补充观察" for item in at.subheader)
    assert any(0 < item.value < 60 for item in at.progress)
    st.cache_resource.clear()
