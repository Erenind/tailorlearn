from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from tailorlearn.db import Repository
from tailorlearn.models import AIUnit, ExtractedSection, PageResult
from tailorlearn.paths import AppPaths
from tailorlearn.providers import ProviderStore


def test_streamlit_restores_provider_selects_new_material_and_selects_all(
    tmp_path: Path,
    monkeypatch,
):
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()

    textbook = repository.create_material_set(
        name="测试课本",
        material_type="textbook",
    )
    standard = repository.create_material_set(
        name="测试课程标准",
        material_type="curriculum_standard",
    )
    document, _ = repository.add_document(
        document_id="doc_test",
        material_set_id=textbook["id"],
        file_name="slice.pdf",
        sha256="abc",
        stored_path="materials/test/source/original.pdf",
        order_index=1,
        page_count=1,
    )
    repository.replace_pages(
        document["id"],
        [
            PageResult(
                page_number=1,
                raw_text="测试内容",
                final_text="测试内容",
                source_kind="text",
            )
        ],
    )
    section = ExtractedSection(
        id="doc_test_p001_s001",
        document_id=document["id"],
        path="测试章节",
        title="测试章节",
        level=1,
        order_index=1,
        page_start=1,
        page_end=1,
        content="【第 1 页】\n测试内容",
    )
    repository.replace_sections(document["id"], [section])
    repository.save_units_for_section(
        document_id=document["id"],
        section_id=section.id,
        units=[
            AIUnit(
                unit_type="concept",
                title="单元一",
                content="内容一",
                source_refs=[],
            ),
            AIUnit(
                unit_type="concept",
                title="单元二",
                content="内容二",
                source_refs=[],
            ),
        ],
        ai_profile="Test",
        ai_base_url="https://example.test/v1",
        ai_model="test-model",
        raw_response_path="test.raw",
    )

    provider_store = ProviderStore(paths.provider_file)
    profile = provider_store.upsert(
        profile_id=None,
        vendor_name="测试厂商",
        base_url="https://example.test/v1",
        model="saved-model",
        api_key="test-key",
    )
    provider_store.set_active(profile.id)

    monkeypatch.setattr(
        AppPaths,
        "default",
        classmethod(lambda cls: paths),
    )
    st.cache_resource.clear()
    app_file = Path(__file__).resolve().parents[1] / "src" / "tailorlearn" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=20).run()

    assert not at.exception
    at.button(key="nav_ai_settings").click().run()
    assert not at.exception
    assert at.selectbox(key="ai_provider_selector").value == profile.id
    assert at.text_input(key="provider_vendor_name").value == "测试厂商"
    assert at.text_input(key="provider_base_url").value == "https://example.test/v1"
    assert at.text_input(key="provider_model").value == "saved-model"
    assert at.text_input(key="provider_api_key").value == ""
    saved_key_display = next(
        item for item in at.text_input if item.label == "已保存的密钥"
    )
    assert "••••" in saved_key_display.value
    assert any("已安全保存" in item.value for item in at.caption)
    assert "test-key" not in repr(at.caption)

    at.button(key="nav_material_center").click().run()
    at.button(key="material_nav_textbook").click().run()
    assert not any(item.label == "课本材料库" for item in at.subheader)
    assert not any("搜索课本材料" == item.label for item in at.text_input)
    at.button(key=f"material_open_{textbook['id']}").click().run()
    at.button(key=f"material_document_open_{document['id']}").click().run()
    assert not at.exception
    assert any(item.value.startswith("### 单元") for item in at.markdown)
    at.segmented_control(
        key="material_section_textbook"
    ).set_value("导入").run()
    assert any(item.label == "课本名称" for item in at.text_input)
    append_checkbox = next(
        item for item in at.checkbox if item.label == "追加到已有课本"
    )
    append_checkbox.set_value(True).run()
    assert not any(item.label == "课本名称" for item in at.text_input)
    at.segmented_control(
        key="material_section_textbook"
    ).set_value("编辑").run()
    at.session_state["_pending_material_library_id"] = textbook["id"]
    at.session_state["_material_library_notice"] = "测试导入完成"
    at.run()
    assert not at.exception
    assert at.selectbox(key="material_library_selector").value == textbook["id"]
    assert any("测试导入完成" in item.value for item in at.success)

    at.button(key="select_all_doc_test").click().run()
    assert not at.exception
    assert any(
        "已全选当前筛选结果，共 2 个单元" in item.value
        for item in at.caption
    )
    assert at.button(key="clear_all_doc_test").label == "取消全选"

    at.button(key="clear_all_doc_test").click().run()
    assert not at.exception
    assert any("已选择 0 个单元" in item.value for item in at.caption)
    st.cache_resource.clear()
