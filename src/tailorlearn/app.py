from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from tailorlearn.ai import AIRequestError, OpenAICompatibleClient
from tailorlearn.db import Repository
from tailorlearn.extraction import page_as_png
from tailorlearn.ingest import IngestionService
from tailorlearn.learning_data import LearningDataService
from tailorlearn.learning_data_ui import render_learning_data_page
from tailorlearn.models import MaterialType, ProviderProfile, UnitStatus
from tailorlearn.paths import AppPaths
from tailorlearn.providers import ProviderStore


st.set_page_config(
    page_title="TailorLearn",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def app_services() -> tuple[
    AppPaths,
    Repository,
    IngestionService,
    ProviderStore,
    LearningDataService,
]:
    paths = AppPaths.default()
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    return (
        paths,
        repository,
        IngestionService(paths, repository),
        ProviderStore(paths.provider_file),
        LearningDataService(repository, paths.db_dir),
    )


PATHS, REPO, INGEST, PROVIDERS, LEARNING_DATA = app_services()

MATERIAL_LABELS = {
    MaterialType.TEXTBOOK.value: "课本",
    MaterialType.STANDARD.value: "课程标准",
}
STATUS_LABELS = {
    UnitStatus.DRAFT.value: "草稿",
    UnitStatus.NEEDS_REVIEW.value: "待复核",
    UnitStatus.VERIFIED.value: "已确认",
}
PROVIDER_SELECTOR_KEY = "ai_provider_selector"
PROVIDER_FIELD_KEYS = {
    "vendor_name": "provider_vendor_name",
    "base_url": "provider_base_url",
    "model": "provider_model",
    "api_key": "provider_api_key",
}


def _rerun() -> None:
    st.rerun()


def _show_error(exc: Exception) -> None:
    st.error(str(exc))


def _load_provider_into_form(profile: ProviderProfile | None) -> None:
    st.session_state[PROVIDER_FIELD_KEYS["vendor_name"]] = (
        profile.vendor_name if profile else ""
    )
    st.session_state[PROVIDER_FIELD_KEYS["base_url"]] = (
        profile.base_url if profile else "https://api.openai.com/v1"
    )
    st.session_state[PROVIDER_FIELD_KEYS["model"]] = profile.model if profile else ""
    st.session_state[PROVIDER_FIELD_KEYS["api_key"]] = ""


def _on_provider_selected() -> None:
    selected_id = st.session_state[PROVIDER_SELECTOR_KEY]
    selected = (
        next(
            (
                profile
                for profile in PROVIDERS.load().profiles
                if profile.id == selected_id
            ),
            None,
        )
        if selected_id != "__new__"
        else None
    )
    _load_provider_into_form(selected)


def render_ai_settings() -> None:
    st.title("AI 参数设置")
    st.caption("支持多个 OpenAI-compatible 厂商配置；密钥仅保存在本机。")
    notice = st.session_state.pop("_provider_settings_notice", None)
    if notice:
        st.success(notice)
    config = PROVIDERS.load()
    profiles = config.profiles

    options = ["__new__", *[profile.id for profile in profiles]]
    pending_profile_id = st.session_state.pop("_pending_provider_select", None)
    if pending_profile_id in options:
        st.session_state[PROVIDER_SELECTOR_KEY] = pending_profile_id
    elif (
        PROVIDER_SELECTOR_KEY not in st.session_state
        or st.session_state[PROVIDER_SELECTOR_KEY] not in options
    ):
        st.session_state[PROVIDER_SELECTOR_KEY] = (
            config.active_profile_id
            if config.active_profile_id in options
            else (profiles[0].id if profiles else "__new__")
        )
    if pending_profile_id in options:
        pending_profile = next(
            (profile for profile in profiles if profile.id == pending_profile_id),
            None,
        )
        _load_provider_into_form(pending_profile)
    if st.session_state.pop("_pending_clear_provider_key", False):
        st.session_state[PROVIDER_FIELD_KEYS["api_key"]] = ""
    if not all(key in st.session_state for key in PROVIDER_FIELD_KEYS.values()):
        selected_for_defaults = next(
            (
                profile
                for profile in profiles
                if profile.id == st.session_state[PROVIDER_SELECTOR_KEY]
            ),
            None,
        )
        _load_provider_into_form(selected_for_defaults)
    selected_id = st.selectbox(
        "厂商配置",
        options,
        key=PROVIDER_SELECTOR_KEY,
        on_change=_on_provider_selected,
        format_func=lambda value: (
            "＋ 新增厂商"
            if value == "__new__"
            else next(
                (
                    f"{profile.vendor_name} · {profile.model}"
                    + (" · 当前" if profile.id == config.active_profile_id else "")
                    for profile in profiles
                    if profile.id == value
                ),
                value,
            )
        ),
    )
    selected = next(
        (profile for profile in profiles if profile.id == selected_id),
        None,
    )

    with st.form("provider_form"):
        vendor_name = st.text_input(
            "厂商名称",
            key=PROVIDER_FIELD_KEYS["vendor_name"],
            placeholder="例如 OpenAI、DeepSeek、Moonshot",
        )
        base_url = st.text_input(
            "Base URL",
            key=PROVIDER_FIELD_KEYS["base_url"],
            help="填写 OpenAI-compatible API 根地址，不在代码中自动追加 /v1。",
        )
        model = st.text_input(
            "默认模型",
            key=PROVIDER_FIELD_KEYS["model"],
            placeholder="填写服务端支持的模型名称",
        )
        if selected and selected.api_key:
            st.text_input(
                "已保存的密钥",
                value=selected.masked_api_key,
                disabled=True,
            )
        api_key = st.text_input(
            "更新 API Key（留空保留）",
            key=PROVIDER_FIELD_KEYS["api_key"],
            type="password",
            placeholder=(
                selected.masked_api_key
                if selected and selected.api_key
                else "请输入 API Key"
            ),
            help=(
                "已保存的密钥不会回填到浏览器。留空提交会保留当前密钥。"
                if selected
                else "首次创建时必须填写，之后可单独清除。"
            ),
        )
        if selected and selected.api_key:
            st.caption(
                "已安全保存；完整密钥不会回填到浏览器。"
                "上方密码输入框仅用于替换密钥，留空提交会保留当前值。"
            )
        else:
            st.caption("当前尚未保存 API Key。")
        make_active = st.checkbox(
            "保存后设为当前配置",
            value=selected is None or selected.id == config.active_profile_id,
        )
        submitted = st.form_submit_button("保存配置", type="primary")
    if submitted:
        try:
            profile = PROVIDERS.upsert(
                profile_id=selected.id if selected else None,
                vendor_name=vendor_name,
                base_url=base_url,
                model=model,
                api_key=api_key if api_key else None,
            )
            if make_active:
                PROVIDERS.set_active(profile.id)
            st.session_state["_pending_provider_select"] = profile.id
            st.session_state["_provider_settings_notice"] = "厂商配置已保存"
            _rerun()
        except Exception as exc:
            _show_error(exc)

    if selected:
        key_status = "已保存" if selected.api_key else "未设置"
        st.info(
            f"当前配置：{selected.vendor_name} · {selected.model} · "
            f"Key {key_status}"
            + (
                f"（{selected.masked_api_key}）"
                if selected.api_key
                else ""
            )
            + f" · `{selected.base_url}`"
        )
        action_col_1, action_col_2, action_col_3 = st.columns(3)
        if action_col_1.button("设为当前配置", width="stretch"):
            PROVIDERS.set_active(selected.id)
            st.session_state["_provider_settings_notice"] = "当前配置已切换"
            _rerun()
        if action_col_2.button("测试连接", width="stretch"):
            try:
                with st.spinner("正在测试 AI 连接..."):
                    message = OpenAICompatibleClient(selected).test_connection()
                st.success(message)
            except AIRequestError as exc:
                st.error(str(exc))
            except Exception as exc:
                _show_error(exc)
        if action_col_3.button("清除密钥", width="stretch"):
            PROVIDERS.clear_key(selected.id)
            st.session_state["_pending_clear_provider_key"] = True
            st.session_state["_provider_settings_notice"] = (
                "本机保存的 API Key 已清除"
            )
            _rerun()

    with st.expander("本机存储说明"):
        st.code(str(PATHS.provider_file), language=None)
        st.caption(
            "该文件位于被 Git 忽略的 .local/ 目录，权限按当前用户私有设置；"
            "业务数据库只记录实际使用的厂商名、base_url 和模型快照。"
        )


def _active_profile() -> ProviderProfile | None:
    return PROVIDERS.active()


def _process_documents_with_progress(documents: list[dict[str, Any]]) -> None:
    progress = st.progress(0, text="准备处理文档")
    for document_index, document in enumerate(documents, start=1):
        def callback(current: int, total: int, message: str) -> None:
            fraction = (
                (document_index - 1) + (current / max(total, 1))
            ) / len(documents)
            progress.progress(
                min(fraction, 1.0),
                text=f"{document['file_name']} · {message}",
            )

        INGEST.process_document(document["id"], callback)
    progress.progress(1.0, text="文字提取与章节切分完成")


def _import_uploaded(
    *,
    material_type: str,
    name: str,
    files: list[Any],
    subject: str,
    stage: str,
    grade: str,
    semester: str,
    edition: str,
    append_to: str | None,
) -> None:
    if not name.strip():
        raise ValueError("材料名称不能为空")
    if not files:
        raise ValueError("至少选择一个 PDF 文件")
    bad = [file.name for file in files if not file.name.lower().endswith(".pdf")]
    if bad:
        raise ValueError(f"仅支持 PDF：{', '.join(bad)}")
    material, _ = INGEST.create_material_from_bytes(
        name=name,
        material_type=material_type,
        files=[(file.name, file.getvalue()) for file in files],
        subject=subject,
        stage=stage,
        grade=grade,
        semester=semester,
        edition=edition,
        append_to=append_to,
    )
    documents = REPO.list_documents(material["id"])
    _process_documents_with_progress(documents)
    st.success(
        f"已保存 {material['name']} 第 {material['version']} 版，"
        f"共 {len(documents)} 个 PDF"
    )
    st.session_state["_pending_material_library_id"] = material["id"]
    st.session_state["_pending_material_library_document_id"] = documents[-1]["id"]
    st.session_state["_material_library_notice"] = (
        f"已导入 {material['name']}（第 {material['version']} 版），"
        f"共 {len(documents)} 个 PDF。"
    )
    _rerun()


def render_import_tab(material_type: str) -> None:
    label = MATERIAL_LABELS[material_type]
    st.subheader(f"导入{label}")
    st.caption(
        "一次只能导入一个版本或一套材料，但可选择多个 PDF 切片。"
        "同一批次不能混合课本与课程标准。"
    )
    existing = REPO.list_material_sets(material_type=material_type)
    append_id: str | None = None
    append_mode = False
    if existing:
        append_mode = st.checkbox(f"追加到已有{label}", value=False)
    if append_mode:
        selected_existing = st.selectbox(
            "已有材料",
            [item["id"] for item in existing],
            format_func=lambda value: next(
                (
                    f"{item['name']} · 第 {item['version']} 版"
                    for item in existing
                    if item["id"] == value
                ),
                value,
            ),
        )
        append_id = selected_existing

    with st.form(f"import_{material_type}"):
        name = st.text_input(
            f"{label}名称",
            placeholder=(
                "例如：义务教育数学课程标准（2022年版）"
                if material_type == MaterialType.STANDARD
                else "例如：七年级数学课本切片"
            ),
        )
        files = st.file_uploader(
            "PDF 文件",
            type=["pdf"],
            accept_multiple_files=True,
        )
        meta_columns = st.columns(3)
        subject = meta_columns[0].text_input("学科（可留空推测）")
        stage = meta_columns[1].text_input("学段（可留空）")
        grade = meta_columns[2].text_input("年级（可留空）")
        meta_columns_2 = st.columns(3)
        semester = meta_columns_2[0].text_input("学期（可留空）")
        edition = meta_columns_2[1].text_input("教材版本（可留空）")
        submitted = st.form_submit_button("保存并提取文字", type="primary")
    if submitted:
        if append_mode and append_id is None:
            st.error("请选择要追加的已有材料")
        else:
            try:
                with st.status("正在导入材料", expanded=True):
                    _import_uploaded(
                        material_type=material_type,
                        name=name,
                        files=files or [],
                        subject=subject,
                        stage=stage,
                        grade=grade,
                        semester=semester,
                        edition=edition,
                        append_to=append_id,
                    )
            except Exception as exc:
                _show_error(exc)

def _material_label(materials: list[dict[str, Any]], material_id: str) -> str:
    material = next(item for item in materials if item["id"] == material_id)
    return (
        f"{MATERIAL_LABELS[material['material_type']]} · "
        f"{material['name']} · 第 {material['version']} 版"
    )


def _document_label(documents: list[dict[str, Any]], document_id: str) -> str:
    document = next(item for item in documents if item["id"] == document_id)
    return f"{document['order_index']}. {document['file_name']} · {document['extraction_status']}"


def render_material_library() -> None:
    st.subheader("材料库")
    materials = REPO.list_material_sets()
    if not materials:
        st.info("尚未导入材料。请先在“导入课本”或“导入课程标准”中创建材料。")
        return
    material_ids = [item["id"] for item in materials]
    pending_material_id = st.session_state.pop("_pending_material_library_id", None)
    if pending_material_id in material_ids:
        st.session_state["material_library_selector"] = pending_material_id
    elif (
        "material_library_selector" not in st.session_state
        or st.session_state["material_library_selector"] not in material_ids
    ):
        st.session_state["material_library_selector"] = material_ids[0]
    notice = st.session_state.pop("_material_library_notice", None)
    if notice:
        st.success(notice)
    material_id = st.selectbox(
        "材料",
        material_ids,
        key="material_library_selector",
        format_func=lambda value: _material_label(materials, value),
    )
    material = REPO.get_material_set(material_id)
    if material is None:
        st.error("材料不存在")
        return
    with st.expander("材料元数据", expanded=False):
        with st.form("material_metadata"):
            columns = st.columns(3)
            subject = columns[0].text_input("学科", value=material["subject"] or "")
            stage = columns[1].text_input("学段", value=material["stage"] or "")
            grade = columns[2].text_input("年级", value=material["grade"] or "")
            columns_2 = st.columns(3)
            semester = columns_2[0].text_input("学期", value=material["semester"] or "")
            edition = columns_2[1].text_input("教材版本", value=material["edition"] or "")
            notes = columns_2[2].text_input("备注", value=material["notes"] or "")
            if st.form_submit_button("保存元数据"):
                REPO.update_material_metadata(
                    material_id,
                    subject=subject,
                    stage=stage,
                    grade=grade,
                    semester=semester,
                    edition=edition,
                    notes=notes,
                )
                st.success("元数据已保存")
                _rerun()
        st.caption("空字段表示尚未确认，不用默认值代替推测结果。")
        try:
            metadata_sources = json.loads(
                material.get("metadata_sources_json") or "{}"
            )
        except json.JSONDecodeError:
            metadata_sources = {}
        if metadata_sources:
            st.caption(
                "字段来源："
                + "；".join(
                    f"{field}={source}"
                    for field, source in metadata_sources.items()
                )
            )

    documents = REPO.list_documents(material_id)
    if not documents:
        st.warning("该材料没有文档")
        return
    document_ids = [item["id"] for item in documents]
    pending_document_id = st.session_state.pop(
        "_pending_material_library_document_id",
        None,
    )
    if pending_document_id in document_ids:
        st.session_state["material_library_document_selector"] = pending_document_id
    elif (
        "material_library_document_selector" not in st.session_state
        or st.session_state["material_library_document_selector"] not in document_ids
    ):
        st.session_state["material_library_document_selector"] = document_ids[0]
    document_id = st.selectbox(
        "文档切片",
        document_ids,
        key="material_library_document_selector",
        format_func=lambda value: _document_label(documents, value),
    )
    document = next(item for item in documents if item["id"] == document_id)
    pages = REPO.list_pages(document_id)
    sections = REPO.list_sections(document_id)
    units = REPO.list_units(document_id=document_id)
    warnings = [page for page in pages if page["status"] != "ok"]

    metrics = st.columns(5)
    metrics[0].metric("页数", len(pages) or document["page_count"])
    metrics[1].metric("章节块", len(sections))
    metrics[2].metric("结构化单元", len(units))
    metrics[3].metric("OCR 异常页", len(warnings))
    metrics[4].metric("提取状态", document["extraction_status"])

    action_columns = st.columns(3)
    if action_columns[0].button("提取/继续切分", width="stretch"):
        try:
            with st.status("正在提取文字并切分", expanded=True):
                progress = st.progress(0.0)

                def callback(current: int, total: int, message: str) -> None:
                    progress.progress(
                        current / max(total, 1),
                        text=message,
                    )

                INGEST.process_document(document_id, callback)
            st.success("提取与切分完成")
            _rerun()
        except Exception as exc:
            _show_error(exc)

    profile = _active_profile()
    structure_disabled = not sections or profile is None
    if action_columns[1].button(
        "开始 AI 结构化",
        type="primary",
        width="stretch",
        disabled=structure_disabled,
    ):
        if profile is None:
            st.error("请先配置当前 AI 厂商")
        else:
            try:
                with st.status(
                    f"正在使用 {profile.vendor_name} / {profile.model} 结构化",
                    expanded=True,
                ):
                    progress = st.progress(0.0)

                    def callback(current: int, total: int, title: str) -> None:
                        progress.progress(
                            current / max(total, 1),
                            text=f"{current}/{total} · {title}",
                        )

                    result = INGEST.structure_document(
                        document_id=document_id,
                        profile=profile,
                        progress=callback,
                    )
                if result["failed"]:
                    st.warning(
                        f"完成 {result['completed']} 块，失败 {result['failed']} 块"
                    )
                    for error in result["errors"][:5]:
                        st.code(error, language=None)
                else:
                    st.success(
                        f"结构化完成：{result['units']} 个单元，运行 {result['run_id']}"
                    )
                _rerun()
            except Exception as exc:
                _show_error(exc)
    if action_columns[2].button("刷新状态", width="stretch"):
        _rerun()
    if not sections:
        st.info("请先完成文字提取和章节切分，再调用 AI。")
    elif profile is None:
        st.warning("尚未设置当前 AI 厂商，无法开始结构化。")

    with st.expander(f"查看章节切分（{len(sections)}）"):
        if sections:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "页码": f"{item['page_start']}-{item['page_end']}",
                            "标题": item["title"],
                            "层级": item["level"],
                            "切分方式": item["split_method"],
                            "ID": item["id"],
                        }
                        for item in sections
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    with st.expander(f"检查 OCR 异常页（{len(warnings)}）", expanded=bool(warnings)):
        warning_pages = [page for page in pages if page["status"] != "ok"]
        if warning_pages:
            page_number = st.selectbox(
                "异常页",
                [page["page_number"] for page in warning_pages],
                format_func=lambda value: next(
                    (
                        f"第 {page['page_number']} 页 · "
                        f"{page['warning_reason'] or page['status']}"
                        for page in warning_pages
                        if page["page_number"] == value
                    ),
                    str(value),
                ),
            )
            page = next(
                item for item in warning_pages if item["page_number"] == page_number
            )
            col_image, col_text = st.columns([1, 2])
            source_path = PATHS.db_dir / document["stored_path"]
            try:
                col_image.image(
                    page_as_png(source_path, page_number),
                    caption=f"第 {page_number} 页原图",
                )
            except Exception as exc:
                col_image.warning(str(exc))
            edited_text = col_text.text_area(
                "识别文本",
                value=page["final_text"],
                height=360,
                key=f"page_text_{document_id}_{page_number}",
            )
            if col_text.button("保存页面文本", key=f"save_page_{page_number}"):
                REPO.update_page_text(document_id, page_number, edited_text)
                if not REPO.list_units(document_id=document_id):
                    INGEST.rebuild_sections(document_id)
                    st.success("页面文本已保存，并已用新文本重建章节切分")
                else:
                    st.success(
                        "页面文本已保存。重试所选单元时会使用最新页面文本；"
                        "章节边界本身不会自动改写。"
                    )
                _rerun()
        else:
            st.success("没有低置信度、空白或识别失败页。")

    st.divider()
    st.subheader("逐单元审核")
    if not units:
        st.info("尚未生成结构化单元。")
        return
    filter_columns = st.columns(2)
    status_filter = filter_columns[0].multiselect(
        "审核状态",
        [status.value for status in UnitStatus],
        default=[status.value for status in UnitStatus],
        format_func=lambda value: STATUS_LABELS.get(value, value),
    )
    section_options = list(
        dict.fromkeys(unit["section_id"] for unit in units)
    )
    section_filter = filter_columns[1].multiselect(
        "章节",
        section_options,
        default=section_options,
        format_func=lambda value: next(
            (
                unit["section_title"]
                for unit in units
                if unit["section_id"] == value
            ),
            value,
        ),
    )
    visible = [
        unit
        for unit in units
        if unit["status"] in status_filter and unit["section_id"] in section_filter
    ]
    if not visible:
        st.info("当前筛选条件下没有单元。")
        return
    table_rows = []
    for unit in visible:
        refs = json.loads(unit["source_refs_json"])
        table_rows.append(
            {
                "ID": unit["id"],
                "类型": unit["unit_type"],
                "标题": unit["title"],
                "章节": unit["section_title"],
                "状态": STATUS_LABELS.get(unit["status"], unit["status"]),
                "页码": ", ".join(str(ref.get("page")) for ref in refs),
                "版本": unit["current_version"],
            }
        )
    visible_ids = [unit["id"] for unit in visible]
    scope_signature = (
        f"{document_id}|{','.join(status_filter)}|{','.join(section_filter)}"
    )
    if st.session_state.get("unit_select_all_scope") != scope_signature:
        st.session_state["unit_select_all_scope"] = scope_signature
        st.session_state["unit_select_all_ids"] = []
    selection_key = f"unit_selection_{document_id}"
    if st.session_state.pop("_clear_unit_table_selection", None) == selection_key:
        st.session_state.pop(selection_key, None)
    select_columns = st.columns([1.2, 1.2, 3])
    if select_columns[0].button(
        "全选当前筛选结果",
        key=f"select_all_{document_id}",
        width="stretch",
    ):
        st.session_state["unit_select_all_ids"] = list(visible_ids)
        st.rerun()
    if select_columns[1].button(
        "取消全选",
        key=f"clear_all_{document_id}",
        width="stretch",
    ):
        st.session_state["unit_select_all_ids"] = []
        st.session_state["_clear_unit_table_selection"] = selection_key
        st.rerun()
    select_columns[2].caption(
        "点击表格行首复选框进行多选；全选覆盖当前筛选结果。"
    )
    table = pd.DataFrame(table_rows)
    selection_event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="multi-row",
        key=selection_key,
        column_order=("ID", "类型", "标题", "章节", "状态", "页码", "版本"),
        alt="结构化单元审核列表",
    )
    selected_rows = list(selection_event.selection.rows)
    manual_ids = [
        visible[index]["id"]
        for index in selected_rows
        if 0 <= index < len(visible)
    ]
    selected_all_ids = [
        unit_id
        for unit_id in st.session_state.get("unit_select_all_ids", [])
        if unit_id in visible_ids
    ]
    selected_ids = list(dict.fromkeys([*manual_ids, *selected_all_ids]))
    if selected_all_ids:
        st.caption(
            f"已全选当前筛选结果，共 {len(selected_ids)} 个单元。"
        )
    else:
        st.caption(f"已选择 {len(selected_ids)} 个单元。")
    if selected_ids:
        button_columns = st.columns(5)
        if button_columns[0].button("标记已确认", width="stretch"):
            count = REPO.set_units_status(selected_ids, UnitStatus.VERIFIED.value)
            INGEST.sync_material_exports(material_id)
            st.success(f"已确认 {count} 个单元")
            _rerun()
        if button_columns[1].button("标记待复核", width="stretch"):
            count = REPO.set_units_status(
                selected_ids,
                UnitStatus.NEEDS_REVIEW.value,
            )
            INGEST.sync_material_exports(material_id)
            st.success(f"已标记 {count} 个单元为待复核")
            _rerun()
        if button_columns[2].button("标记草稿", width="stretch"):
            count = REPO.set_units_status(selected_ids, UnitStatus.DRAFT.value)
            INGEST.sync_material_exports(material_id)
            st.success(f"已恢复 {count} 个单元为草稿")
            _rerun()
        if button_columns[3].button("重新提交 AI", width="stretch"):
            if profile is None:
                st.error("请先配置当前 AI 厂商")
            else:
                try:
                    with st.spinner("正在重试所选章节..."):
                        result = INGEST.retry_units(
                            unit_ids=selected_ids,
                            profile=profile,
                        )
                    if result["failed"]:
                        st.warning(
                            f"成功 {result['completed']} 个章节，"
                            f"失败 {result['failed']} 个"
                        )
                        for error in result["errors"]:
                            st.code(error, language=None)
                    else:
                        st.success(f"已重试 {result['completed']} 个章节")
                    _rerun()
                except Exception as exc:
                    _show_error(exc)
        if button_columns[4].button("软删除", width="stretch"):
            count = REPO.soft_delete_units(selected_ids)
            INGEST.sync_material_exports(material_id)
            st.warning(f"已软删除 {count} 个单元，可从已删除列表恢复")
            _rerun()

    unit_id = st.selectbox(
        "查看或编辑单个单元",
        [unit["id"] for unit in visible],
        format_func=lambda value: next(
            (unit["title"] for unit in visible if unit["id"] == value),
            value,
        ),
    )
    unit = next(item for item in visible if item["id"] == unit_id)
    refs = json.loads(unit["source_refs_json"])
    with st.form(f"edit_unit_{unit_id}"):
        edit_columns = st.columns([3, 1])
        edited_title = edit_columns[0].text_input("标题", value=unit["title"])
        edited_type = edit_columns[1].text_input("类型", value=unit["unit_type"])
        edited_content = st.text_area(
            "忠实整理文本",
            value=unit["content"],
            height=280,
        )
        st.caption(
            "来源："
            + "；".join(
                f"第 {ref.get('page')} 页 {ref.get('quote', '')[:60]}"
                for ref in refs
            )
        )
        if st.form_submit_button("保存为新版本"):
            REPO.update_unit(
                unit_id=unit_id,
                title=edited_title,
                unit_type=edited_type,
                content=edited_content,
            )
            INGEST.sync_material_exports(material_id)
            st.success("已保存为新版本并回到草稿状态")
            _rerun()

    deleted_units = REPO.list_units(
        document_id=document_id,
        include_deleted=True,
    )
    deleted_units = [item for item in deleted_units if item["deleted_at"]]
    with st.expander(f"已删除单元（{len(deleted_units)}）"):
        if not deleted_units:
            st.caption("没有软删除的单元")
        else:
            restore_options = st.multiselect(
                "选择要恢复的单元",
                [item["id"] for item in deleted_units],
                format_func=lambda value: next(
                    (
                        f"{item['title']} · {item['id']}"
                        for item in deleted_units
                        if item["id"] == value
                    ),
                    value,
                ),
            )
            if st.button("恢复所选单元", disabled=not restore_options):
                count = REPO.restore_units(restore_options)
                INGEST.sync_material_exports(material_id)
                st.success(f"已恢复 {count} 个单元")
                _rerun()


def render_subject_analysis() -> None:
    st.title("学科分析")
    st.caption("结构化导入阶段不生成最终知识点，也不建立跨材料联系。")
    import_textbook, import_standard, library = st.tabs(
        ["导入课本", "导入课程标准", "材料库与审核"]
    )
    with import_textbook:
        render_import_tab(MaterialType.TEXTBOOK.value)
    with import_standard:
        render_import_tab(MaterialType.STANDARD.value)
    with library:
        render_material_library()


def main() -> None:
    st.sidebar.title("TailorLearn")
    page = st.sidebar.radio(
        "导航",
        ["AI 参数设置", "学科分析", "学情数据"],
        key="main_navigation",
    )
    st.sidebar.caption("本轮不含智能对话入口")
    if page == "AI 参数设置":
        render_ai_settings()
    elif page == "学科分析":
        render_subject_analysis()
    else:
        render_learning_data_page(LEARNING_DATA)


main()
