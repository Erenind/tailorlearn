from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from tailorlearn.analytics_ui import _student_code_from_click
from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.paths import AppPaths


def test_student_code_from_button_column_click() -> None:
    assert _student_code_from_click(
        {"row": 1},
        ["D01", "D02"],
    ) == "D02"
    assert _student_code_from_click(None, ["D01"]) is None
    assert _student_code_from_click({"row": 2}, ["D01"]) is None
    assert _student_code_from_click(
        {"row": "invalid"},
        ["D01"],
    ) is None


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
    assert "教学智能体" in at.button(key="nav_teaching_agent").label
    assert "教师工作台" in at.button(key="nav_workbench").label
    at.button(key="nav_workbench").click().run()
    assert not at.exception
    assert any(item.value == "教师工作台" for item in at.title)
    assert any(item.value == "待办概览" for item in at.subheader)
    at.button(key="nav_student").click().run()
    assert not at.exception
    assert any(item.value == "学生分析" for item in at.title)
    assert at.button(
        key=f"analytics_student_dataset_open_{dataset_id}"
    ).label == "进入分析"
    assert not any(
        item.key == "student_analysis_nav_overview" for item in at.button
    )
    assert not any(item.value == "数据统计标签" for item in at.subheader)

    at.button(key=f"analytics_student_dataset_open_{dataset_id}").click().run()
    assert not at.exception
    assert at.button(key="analytics_student_path_datasets").label == "学情数据集"
    ranking = next(
        item.value
        for item in at.dataframe
        if "综合名次" in item.value.columns
    )
    assert ranking.loc[0, "综合名次"] == "1"
    assert ranking.loc[0, "学生编号"] == "D01"
    assert ranking.loc[0, "综合得分率"] == 1.0
    assert "D06" not in set(ranking["学生编号"])
    incomplete = next(
        item.value
        for item in at.dataframe
        if "已观察主题得分率" in item.value.columns
    )
    assert incomplete.loc[0, "学生编号"] == "D06"
    assert incomplete.loc[0, "记录完整度"] == "0/1（0%）"
    assert incomplete.loc[0, "已观察主题"] == "1/1"
    assert "查看" in ranking.columns
    assert "查看" in incomplete.columns
    assert not any(
        item.key == "student_analysis_nav_overview" for item in at.button
    )
    assert at.session_state.get("analytics_student_detail_code") is None

    at.session_state["analytics_student_detail_code"] = "D01"
    at.run()
    assert not at.exception
    assert at.button(key="student_analysis_nav_overview").label.endswith("数据概览")
    assert at.button(key="student_analysis_nav_issues").label.endswith("问题分析")
    assert any(item.value == "数据统计标签" for item in at.subheader)
    assert at.button(
        key=f"analytics_student_back_to_rankings_{dataset_id}"
    ).label == "返回榜单"

    at.button(
        key=f"analytics_student_back_to_rankings_{dataset_id}"
    ).click().run()
    assert not at.exception
    assert next(
        item.value
        for item in at.dataframe
        if "综合名次" in item.value.columns
    ).loc[0, "学生编号"] == "D01"
    assert not any(
        item.key == "student_analysis_nav_overview" for item in at.button
    )

    at.session_state["analytics_student_detail_code"] = "D06"
    at.run()
    assert not at.exception
    assert any(
        "学生 D06" in item.value and "暂不参与综合排名" in item.value
        for item in at.caption
    )
    assert at.session_state.get("analytics_student_detail_code") == "D06"

    at.button(
        key=f"analytics_student_back_to_rankings_{dataset_id}"
    ).click().run()
    at.button(key="analytics_student_path_datasets").click().run()
    assert not at.exception
    assert at.button(
        key=f"analytics_student_dataset_open_{dataset_id}"
    ).label == "进入分析"
    at.button(key=f"analytics_student_dataset_open_{dataset_id}").click().run()
    at.session_state["analytics_student_detail_code"] = "D01"
    at.run()

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
