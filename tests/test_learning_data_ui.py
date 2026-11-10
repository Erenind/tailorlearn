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
    at = AppTest.from_file(str(app_file), default_timeout=20).run()

    assert not at.exception
    at.button(key="nav_material_center").click().run()
    at.button(key="material_nav_students").click().run()
    at.segmented_control(
        key="material_section_students"
    ).set_value("导入").run()
    assert not at.exception
    assert at.selectbox(key="learning_dataset_selector").value == dataset["id"]

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
    ).options == ["浏览", "编辑", "导入", "异常检测"]
    at.segmented_control(
        key="material_section_answers"
    ).set_value("异常检测").run()
    assert not at.exception
    assert any(
        item.value == "作答记录异常检测"
        for item in at.subheader
    )
    st.cache_resource.clear()
