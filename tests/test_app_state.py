from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from tailorlearn.db import Repository
from tailorlearn.ingest import IngestionService
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
    workspace_html = next(
        item.value for item in at.html if "tl-material-workspace" in item.value
    )
    assert "单元一" in workspace_html
    assert 'href="#tl-unit-' in workspace_html
    assert "height: calc(100vh - 20rem)" in workspace_html
    assert workspace_html.count("height: 100%") >= 2
    assert "tl-directory-pane" in workspace_html
    assert "tl-content-pane" in workspace_html
    at.segmented_control(
        key="material_section_textbook"
    ).set_value("导入和导出").run()
    assert any(item.label == "课本名称" for item in at.text_input)
    append_checkbox = next(
        item for item in at.checkbox if item.label == "追加到已有课本"
    )
    append_checkbox.set_value(True).run()
    assert not any(item.label == "课本名称" for item in at.text_input)
    at.segmented_control(
        key="material_section_textbook"
    ).set_value("AI结构化和管理").run()
    assert not at.exception
    at.button(key=f"material_manage_open_{textbook['id']}").click().run()
    at.button(key=f"material_manage_open_{document['id']}").click().run()
    assert not at.exception
    units = repository.list_units(document_id=document["id"])
    assert len(units) == 2
    assert at.button(key="material_reprocess_doc_test").label == "重新识别"
    assert at.button(key=f"material_edit_unit_{units[0]['id']}").label == "编辑"
    assert at.button(key=f"material_delete_unit_{units[0]['id']}").label == "删除"

    at.button(key="material_select_all_doc_test").click().run()
    assert not at.exception
    for unit in units:
        checkbox = at.checkbox(
            key=(
                f"material_manage_directory_select_doc_test_"
                f"{unit['id']}"
            )
        )
        assert checkbox.value is True

    at.button(key="material_confirm_selected_doc_test").click().run()
    assert not at.exception
    assert all(
        unit["status"] == "verified"
        for unit in repository.list_units(document_id=document["id"])
    )
    assert at.button(key="material_ai_structure_doc_test").label == (
        "重新 AI 结构化"
    )
    at.button(key="material_select_all_doc_test").click().run()
    at.button(key="material_delete_selected_doc_test").click().run()
    assert all(
        unit["deleted_at"] is not None
        for unit in repository.list_units(
            document_id=document["id"],
            include_deleted=True,
        )
    )
    at.button(key="material_trash_doc_test").click().run()
    assert any(item.value == "回收站" for item in at.subheader)
    st.cache_resource.clear()


def test_streamlit_browses_and_edits_standard_sections(
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
    document, _ = repository.add_document(
        document_id="doc_standard",
        material_set_id=standard["id"],
        file_name="标准.pdf",
        sha256="standard",
        stored_path="materials/standard/source/original.pdf",
        order_index=1,
        page_count=2,
    )
    parent = ExtractedSection(
        id="doc_standard_p001_s001",
        document_id=document["id"],
        path="课程内容",
        title="课程内容",
        level=1,
        order_index=1,
        page_start=1,
        page_end=2,
        content="课程内容正文",
    )
    child = ExtractedSection(
        id="doc_standard_p001_s002",
        document_id=document["id"],
        parent_id=parent.id,
        path="课程内容 / 数与代数",
        title="数与代数",
        level=2,
        order_index=2,
        page_start=1,
        page_end=1,
        content="数与代数正文",
    )
    repository.replace_sections(document["id"], [parent, child])

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
    at.button(key="material_nav_standard").click().run()
    section_control = at.segmented_control(key="material_section_standard")
    assert section_control.options == ["浏览", "编辑", "导入和导出"]

    at.button(key=f"material_open_{standard['id']}").click().run()
    at.button(key=f"material_document_open_{document['id']}").click().run()
    assert not at.exception
    workspace_html = next(
        item.value for item in at.html if "tl-material-workspace" in item.value
    )
    assert "课程内容" in workspace_html
    assert "数与代数" in workspace_html
    assert "第 1–2 页" in workspace_html
    assert "前往编辑并开始结构化" not in workspace_html

    section_control = at.segmented_control(key="material_section_standard")
    section_control.set_value("编辑").run()
    assert not any("AI结构化" in button.label for button in at.button)
    at.button(key=f"standard_editor_open_{standard['id']}").click().run()
    at.button(
        key=f"standard_editor_document_open_{document['id']}"
    ).click().run()
    at.button(key=f"standard_editor_edit_{parent.id}").click().run()
    at.text_input(
        key=f"standard_editor_title_{parent.id}"
    ).set_value("课程内容（修订）")
    at.text_area(
        key=f"standard_editor_content_{parent.id}"
    ).set_value("修订后的课程标准正文")
    at.button(key=f"standard_editor_save_{parent.id}").click().run()

    assert not at.exception
    saved_sections = repository.list_sections(document["id"])
    assert saved_sections[0]["title"] == "课程内容（修订）"
    assert saved_sections[0]["content"] == "修订后的课程标准正文"
    assert saved_sections[1]["path"] == "课程内容（修订） / 数与代数"
    assert any("已保存" in item.value for item in at.success)
    st.cache_resource.clear()


def test_streamlit_deletes_slices_from_material_management(
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
    textbook_document, _ = repository.add_document(
        document_id="doc_textbook",
        material_set_id=textbook["id"],
        file_name="课本切片.pdf",
        sha256="textbook",
        stored_path="materials/textbook.pdf",
        order_index=1,
        page_count=1,
    )
    standard = repository.create_material_set(
        name="测试课程标准",
        material_type="curriculum_standard",
    )
    standard_document, _ = repository.add_document(
        document_id="doc_standard_delete",
        material_set_id=standard["id"],
        file_name="课标切片.pdf",
        sha256="standard-delete",
        stored_path="materials/standard.pdf",
        order_index=1,
        page_count=1,
    )

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
    at.segmented_control(
        key="material_section_textbook"
    ).set_value("AI结构化和管理").run()
    at.button(key=f"material_manage_open_{textbook['id']}").click().run()
    assert at.button(
        key=f"material_manage_delete_slice_{textbook_document['id']}"
    ).label == "删除切片"
    at.button(
        key=f"material_manage_delete_slice_{textbook_document['id']}"
    ).click().run()
    assert any("确认删除切片" in item.value for item in at.warning)
    at.button(
        key=f"material_manage_delete_slice_confirm_delete_{textbook_document['id']}"
    ).click().run()
    assert repository.list_documents(textbook["id"]) == []

    at.button(key="material_nav_standard").click().run()
    at.segmented_control(
        key="material_section_standard"
    ).set_value("编辑").run()
    at.button(key=f"standard_editor_open_{standard['id']}").click().run()
    assert at.button(
        key=f"standard_editor_delete_slice_{standard_document['id']}"
    ).label == "删除切片"
    at.button(
        key=f"standard_editor_delete_slice_{standard_document['id']}"
    ).click().run()
    at.button(
        key=f"standard_editor_delete_slice_confirm_delete_{standard_document['id']}"
    ).click().run()
    assert repository.list_documents(standard["id"]) == []
    st.cache_resource.clear()


def test_streamlit_reimport_restores_deleted_textbook_slice(
    tmp_path: Path,
    sample_pdf: Path,
    monkeypatch,
):
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    service = IngestionService(paths, repository)
    material, documents = service.create_material_from_paths(
        name="重新导入测试课本",
        material_type="textbook",
        pdf_paths=[sample_pdf],
    )
    document = documents[0]
    service.process_document(document["id"])
    assert repository.soft_delete_document(document["id"]) is True

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
    at.segmented_control(
        key="material_section_textbook"
    ).set_value("导入和导出").run()
    append_checkbox = next(
        item for item in at.checkbox if item.label == "追加到已有课本"
    )
    append_checkbox.set_value(True).run()
    at.file_uploader[0].upload("sample.pdf", sample_pdf.read_bytes())
    next(
        button
        for button in at.form_submit_button
        if button.label == "保存并提取文字"
    ).click().run()

    assert not at.exception
    restored = repository.list_documents(material["id"])
    assert [item["id"] for item in restored] == [document["id"]]
    at.segmented_control(
        key="material_section_textbook"
    ).set_value("AI结构化和管理").run()
    at.button(key=f"material_manage_open_{material['id']}").click().run()
    assert at.button(key=f"material_manage_open_{document['id']}").label == (
        "查看切片"
    )
    st.cache_resource.clear()


def test_streamlit_renames_slices_in_both_management_views(
    tmp_path: Path,
    monkeypatch,
):
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()

    textbook = repository.create_material_set(
        name="重命名测试课本",
        material_type="textbook",
    )
    textbook_document, _ = repository.add_document(
        document_id="doc_rename_textbook",
        material_set_id=textbook["id"],
        file_name="原课本切片.pdf",
        sha256="rename-textbook",
        stored_path="materials/textbook/source/original.pdf",
        order_index=1,
        page_count=1,
    )
    standard = repository.create_material_set(
        name="重命名测试课程标准",
        material_type="curriculum_standard",
    )
    standard_document, _ = repository.add_document(
        document_id="doc_rename_standard",
        material_set_id=standard["id"],
        file_name="原课标切片.pdf",
        sha256="rename-standard",
        stored_path="materials/standard/source/original.pdf",
        order_index=1,
        page_count=1,
    )

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
    at.segmented_control(
        key="material_section_textbook"
    ).set_value("AI结构化和管理").run()
    at.button(key=f"material_manage_open_{textbook['id']}").click().run()
    at.button(
        key=f"material_manage_rename_slice_{textbook_document['id']}"
    ).click().run()
    at.text_input(
        key=f"material_manage_rename_slice_input_{textbook_document['id']}"
    ).set_value("重命名后的课本切片.pdf")
    at.button(
        key=f"material_manage_rename_slice_save_{textbook_document['id']}"
    ).click().run()
    assert not at.exception
    assert repository.get_document(textbook_document["id"])["file_name"] == (
        "重命名后的课本切片.pdf"
    )

    at.button(key="material_nav_standard").click().run()
    at.segmented_control(
        key="material_section_standard"
    ).set_value("编辑").run()
    at.button(key=f"standard_editor_open_{standard['id']}").click().run()
    at.button(
        key=f"standard_editor_rename_slice_{standard_document['id']}"
    ).click().run()
    at.text_input(
        key=f"standard_editor_rename_slice_input_{standard_document['id']}"
    ).set_value("重命名后的课标切片.pdf")
    at.button(
        key=f"standard_editor_rename_slice_save_{standard_document['id']}"
    ).click().run()
    assert not at.exception
    assert repository.get_document(standard_document["id"])["file_name"] == (
        "重命名后的课标切片.pdf"
    )
    st.cache_resource.clear()
