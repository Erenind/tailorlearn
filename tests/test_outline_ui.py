from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.models import ExtractedSection
from tailorlearn.outline_ui import _section_label, _workflow_rows
from tailorlearn.paths import AppPaths


def test_outline_page_uses_center_content_and_right_settings(
    tmp_path: Path,
    monkeypatch,
):
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()

    standard = repository.create_material_set(
        name="测试课程标准",
        material_type="curriculum_standard",
    )
    repository.add_document(
        document_id="doc_standard_ui",
        material_set_id=standard["id"],
        file_name="standard.pdf",
        sha256="standard-ui",
        stored_path="materials/standard.pdf",
        order_index=1,
        page_count=1,
    )
    repository.replace_sections(
        "doc_standard_ui",
        [
            ExtractedSection(
                id="doc_standard_ui_p001_s001",
                document_id="doc_standard_ui",
                path="课程内容",
                title="课程内容",
                level=1,
                order_index=1,
                page_start=1,
                page_end=1,
                content="课程内容",
            )
        ],
    )
    textbook = repository.create_material_set(
        name="测试课本",
        material_type="textbook",
    )
    repository.add_document(
        document_id="doc_textbook_ui",
        material_set_id=textbook["id"],
        file_name="textbook.pdf",
        sha256="textbook-ui",
        stored_path="materials/textbook.pdf",
        order_index=1,
        page_count=1,
    )
    repository.replace_sections(
        "doc_textbook_ui",
        [
            ExtractedSection(
                id="doc_textbook_ui_p001_s001",
                document_id="doc_textbook_ui",
                path="第一章",
                title="第一章",
                level=1,
                order_index=1,
                page_start=1,
                page_end=1,
                content="第一章",
            )
        ],
    )
    learning = LearningDataService(repository, paths.db_dir)
    dataset = learning.create_dataset("测试学情数据")
    learning.create_record(
        dataset["id"],
        "requirement",
        {
            "theme_code": "T01",
            "theme_name": "测试主题",
            "theme_goal": "测试目标",
            "indicator_code": "T01I1",
            "indicator_name": "测试指标",
            "standard_ref": "CR01",
            "standard_page": "1",
            "textbook_page": "1",
        },
    )

    monkeypatch.setattr(
        AppPaths,
        "default",
        classmethod(lambda cls: paths),
    )
    st.cache_resource.clear()
    app_file = Path(__file__).resolve().parents[1] / "src" / "tailorlearn" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=20).run()
    at.button(key="nav_outline").click().run()

    assert not at.exception
    assert at.container(key="outline_settings_panel")
    mode = at.segmented_control(key="outline_scope_mode")
    assert mode.options == ["课本模式", "主题模式"]
    assert mode.value == "课本模式"
    assert any(
        "课本模式不涉及学情数据" in item.value for item in at.info
    )
    requirements = at.text_area(key="outline_specific_requirements")
    assert requirements.value == ""
    assert any(
        item.label == "具体要求（结构化生成阶段）"
        for item in at.expander
    )

    mode.set_value("主题模式").run()
    assert not at.exception
    assert at.selectbox(key="outline_theme_scope_dataset")
    assert at.multiselect(
        key=f"outline_theme_scope_themes_{dataset['id']}"
    )
    assert at.multiselect(key="outline_theme_scope_textbooks")
    st.cache_resource.clear()


def test_section_label_supports_two_stage_directory_entries():
    label = _section_label(
        {
            "file_name": "教材.pdf",
            "path": "第一章 / 1.1 方程",
            "title": "1.1 方程",
            "page_start": 2,
            "page_end": 4,
        }
    )
    assert "第一章 / 1.1 方程" in label
    assert "第 2–4 页" in label


def test_legacy_workflow_uses_formal_stage_names():
    rows = _workflow_rows(
        [
            "已发送范围与课程标准目录",
            "第一次 AI 返回需要读取的目录",
            "已获取具体内容",
            "第二次 AI 返回格式化结果",
        ]
    )
    assert [row[0] for row in rows] == [
        "范围准备与目录提交",
        "目录检索阶段",
        "内容提取阶段",
        "结构化生成阶段",
    ]
    assert all("第一次" not in detail for _, detail in rows)
    assert all("第二次" not in detail for _, detail in rows)


def test_generated_run_is_queued_without_mutating_instantiated_history_widget():
    def script():
        import streamlit as st

        from tailorlearn.outline_ui import _queue_history_run

        st.selectbox("历史梳理清单", ["old"], key="outline_history_run")
        _queue_history_run("new")
        st.write(st.session_state["_outline_pending_history_run"])

    at = AppTest.from_function(script).run()
    assert not at.exception
    assert at.selectbox(key="outline_history_run").value == "old"
    assert at.session_state["_outline_pending_history_run"] == "new"


def test_pending_history_run_is_applied_before_widget_creation():
    def script():
        import streamlit as st

        from tailorlearn.outline_ui import _apply_pending_history_run

        st.session_state["_outline_pending_history_run"] = "new"
        _apply_pending_history_run()
        st.selectbox("历史梳理清单", ["old", "new"], key="outline_history_run")

    at = AppTest.from_function(script).run()
    assert not at.exception
    assert at.selectbox(key="outline_history_run").value == "new"
