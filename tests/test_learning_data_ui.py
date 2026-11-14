from __future__ import annotations

import csv
import io
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.paths import AppPaths


def test_streamlit_learning_data_upload_preview_and_import(
    tmp_path: Path,
    monkeypatch,
):
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    service = LearningDataService(repository, paths.db_dir)
    dataset = service.create_dataset("页面测试数据集")

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["学生编号", "模拟班级编号", "年级"])
    writer.writerow(["D01", "DC01", "7"])
    content = output.getvalue().encode("utf-8-sig")

    monkeypatch.setattr(
        AppPaths,
        "default",
        classmethod(lambda cls: paths),
    )
    st.cache_resource.clear()
    app_file = Path(__file__).resolve().parents[1] / "src" / "tailorlearn" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=40).run()

    assert not at.exception
    at.button(key="nav_material_center").click().run()
    at.button(key="material_nav_students").click().run()
    at.button(key=f"learning_dataset_open_{dataset['id']}").click().run()
    at.segmented_control(
        key="material_section_students"
    ).set_value("导入和导出").run()
    assert any(item.label == "数据集名称" for item in at.text_input)
    assert not any(
        item.label == "新建学情数据集" for item in at.button
    )
    assert not at.exception
    assert at.button(key="learning_dataset_path_root").label == "学情数据集"

    upload_key = f"learning_upload_{dataset['id']}_student"
    at.file_uploader(key=upload_key).upload("任意文件名.csv", content).run()
    assert not at.exception
    mapping_widgets = [
        widget
        for widget in at.selectbox
        if "learning_mapping" in (widget.key or "")
    ]
    assert [widget.value for widget in mapping_widgets] == [
        "student_code",
        "class_code",
        "grade",
    ]

    confirm_key = f"learning_confirm_import_{dataset['id']}_student"
    at.button(key=confirm_key).click().run()
    assert not at.exception
    assert any("导入完成：新增 1" in item.value for item in at.success)
    rows = service.list_records(dataset["id"], "student")
    assert len(rows) == 1
    assert rows[0]["student_code"] == "D01"

    assert at.button(key="material_nav_textbook").label.endswith("课本")
    assert at.button(key="material_nav_standard").label.endswith("课程标准")
    assert at.button(key="material_nav_answers").label.endswith("作答记录")
    at.button(key="material_nav_answers").click().run()
    assert at.segmented_control(
        key="material_section_answers"
    ).options == ["浏览", "编辑", "导入和导出", "异常检测"]
    at.segmented_control(
        key="material_section_answers"
    ).set_value("异常检测").run()
    assert not at.exception
    assert any(
        item.value == "作答记录异常检测"
        for item in at.subheader
    )
    st.cache_resource.clear()


def test_dataset_creation_is_available_only_in_import(
    tmp_path: Path,
    monkeypatch,
):
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    service = LearningDataService(repository, paths.db_dir)

    monkeypatch.setattr(
        AppPaths,
        "default",
        classmethod(lambda cls: paths),
    )
    st.cache_resource.clear()
    app_file = Path(__file__).resolve().parents[1] / "src" / "tailorlearn" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=40).run()

    assert not at.exception
    at.button(key="nav_material_center").click().run()
    at.button(key="material_nav_students").click().run()
    assert not any(item.label == "数据集名称" for item in at.text_input)
    assert any("尚无学情数据集" in item.value for item in at.info)

    at.segmented_control(
        key="material_section_students"
    ).set_value("编辑").run()
    assert not any(item.label == "数据集名称" for item in at.text_input)

    at.segmented_control(
        key="material_section_students"
    ).set_value("导入和导出").run()
    assert any(item.label == "数据集名称" for item in at.text_input)
    assert any(item.label == "创建数据集" for item in at.button)
    assert not any(
        (button.key or "").startswith("learning_dataset_open_")
        for button in at.button
    )
    assert not at.exception
    st.cache_resource.clear()


def test_dataset_catalog_tables_and_anomaly_actions(
    tmp_path: Path,
    monkeypatch,
):
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    service = LearningDataService(repository, paths.db_dir)
    dataset = service.create_dataset("表格与异常检测数据集")
    service.create_record(
        dataset["id"],
        "student",
        {"student_code": "D01", "class_code": "DC01", "grade": 7},
    )
    service.create_record(
        dataset["id"],
        "answer",
        {
            "record_code": "DR001",
            "student_code": "MISSING",
            "material_code": "MISSING",
            "theme_code": "T99",
            "node_code": "N99",
            "answer_event_code": "EVENT",
            "revision": 1,
            "submitted_at": "2026-01-20T08:00:00+08:00",
            "grading_status": "已评分",
            "score": 4,
            "full_score": 3,
        },
    )

    monkeypatch.setattr(
        AppPaths,
        "default",
        classmethod(lambda cls: paths),
    )
    st.cache_resource.clear()
    app_file = Path(__file__).resolve().parents[1] / "src" / "tailorlearn" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=40).run()

    assert not at.exception
    at.button(key="nav_material_center").click().run()
    at.button(key="material_nav_students").click().run()
    assert not any(item.label == "数据集名称" for item in at.text_input)
    assert at.button(key=f"learning_dataset_open_{dataset['id']}").label == "查看"
    at.segmented_control(
        key="material_section_students"
    ).set_value("导入和导出").run()
    assert any(item.label == "数据集名称" for item in at.text_input)
    assert not any(
        (button.key or "").startswith("learning_dataset_open_")
        for button in at.button
    )
    at.segmented_control(
        key="material_section_students"
    ).set_value("浏览").run()
    at.button(key=f"learning_dataset_open_{dataset['id']}").click().run()
    assert len(at.dataframe) == 1
    assert not any(
        (item.key or "").startswith("learning_data_table_")
        for item in at.container
    )
    assert not any(
        item.key == f"learning_search_{dataset['id']}_student"
        for item in at.text_input
    )
    browse_table = at.dataframe[0].value
    assert "学生编号" in browse_table.columns
    assert "编辑" not in browse_table.columns
    assert "删除" not in browse_table.columns

    at.segmented_control(
        key="material_section_students"
    ).set_value("编辑").run()
    assert not at.exception
    edit_table = at.dataframe[0].value
    assert "编辑" in edit_table.columns
    assert "删除" in edit_table.columns
    assert not any(
        (item.key or "").startswith("learning_data_table_")
        for item in at.container
    )
    assert at.text_input(
        key=f"learning_search_{dataset['id']}_student"
    ).label == "搜索"
    assert any(
        item.label == "显示已删除" and item.value is False
        for item in at.toggle
    )
    assert not any(
        item.value == "数据管理与导出"
        for item in at.subheader
    )
    assert not any(item.label == "数据集名称" for item in at.text_input)
    at.segmented_control(
        key="material_section_students"
    ).set_value("导入和导出").run()
    assert any(item.label == "数据集名称" for item in at.text_input)
    assert not any(
        item.label == "新建学情数据集" for item in at.button
    )

    at.button(key="material_nav_answers").click().run()
    at.segmented_control(
        key="material_section_answers"
    ).set_value("异常检测").run()
    assert not at.exception
    assert {metric.label for metric in at.metric} >= {
        "作答记录",
        "异常项",
        "涉及记录",
        "待评分/未作答",
    }
    assert any(
        {"编辑", "删除"}.issubset(set(frame.value.columns))
        for frame in at.dataframe
    )
    st.cache_resource.clear()
