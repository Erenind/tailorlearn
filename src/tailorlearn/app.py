from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from tailorlearn.analytics import AnalyticsService
from tailorlearn.analytics_ui import (
    render_class_analysis_page,
    render_student_analysis_page,
)
from tailorlearn.ai import AIRequestError, OpenAICompatibleClient
from tailorlearn.db import Repository
from tailorlearn.extraction import page_as_png
from tailorlearn.ingest import IngestionService
from tailorlearn.learning_data import LearningDataService
from tailorlearn.learning_data_ui import (
    render_answer_anomalies,
    render_data_type_page,
    render_dataset_context,
)
from tailorlearn.models import MaterialType, ProviderProfile, UnitStatus
from tailorlearn.outline import OutlineService
from tailorlearn.outline_ui import render_outline_page
from tailorlearn.paths import AppPaths
from tailorlearn.providers import ProviderStore
from tailorlearn.student_diagnosis import StudentDiagnosisService
from tailorlearn.teaching_recommendations import TeachingRecommendationService
from tailorlearn.teaching_recommendations_ui import render_teaching_recommendations_page
from tailorlearn.ui import (
    apply_app_style,
    render_page_header,
    render_sidebar_navigation,
)


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
    OutlineService,
    AnalyticsService,
    StudentDiagnosisService,
    TeachingRecommendationService,
]:
    paths = AppPaths.default()
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    analytics = AnalyticsService(repository)
    return (
        paths,
        repository,
        IngestionService(paths, repository),
        ProviderStore(paths.provider_file),
        LearningDataService(repository, paths.db_dir),
        OutlineService(paths, repository),
        analytics,
        StudentDiagnosisService(repository, analytics),
        TeachingRecommendationService(repository, analytics),
    )


(
    PATHS,
    REPO,
    INGEST,
    PROVIDERS,
    LEARNING_DATA,
    OUTLINES,
    ANALYTICS,
    DIAGNOSES,
    TEACHING_RECOMMENDATIONS,
) = app_services()

MATERIAL_LABELS = {
    MaterialType.TEXTBOOK.value: "课本",
    MaterialType.STANDARD.value: "课程标准",
}
MATERIAL_CENTER_ITEMS = (
    {
        "key": "textbook",
        "label": "课本",
        "icon": "textbook.svg",
        "kind": "material",
        "data_type": MaterialType.TEXTBOOK.value,
    },
    {
        "key": "standard",
        "label": "课程标准",
        "icon": "standard.svg",
        "kind": "material",
        "data_type": MaterialType.STANDARD.value,
    },
    {
        "key": "students",
        "label": "学生",
        "icon": "student.svg",
        "kind": "learning",
        "data_type": "student",
    },
    {
        "key": "requirements",
        "label": "课程要求",
        "icon": "requirements.svg",
        "kind": "learning",
        "data_type": "requirement",
    },
    {
        "key": "materials",
        "label": "题目与材料",
        "icon": "question_material.svg",
        "kind": "learning",
        "data_type": "material",
    },
    {
        "key": "nodes",
        "label": "测验节点",
        "icon": "node.svg",
        "kind": "learning",
        "data_type": "node",
    },
    {
        "key": "answers",
        "label": "作答记录",
        "icon": "answers.svg",
        "kind": "learning",
        "data_type": "answer",
    },
    {
        "key": "fields",
        "label": "字段说明",
        "icon": "field_spec.svg",
        "kind": "learning",
        "data_type": "field_spec",
    },
)
MATERIAL_CENTER_ITEM_BY_KEY = {
    item["key"]: item for item in MATERIAL_CENTER_ITEMS
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

    with st.container(key="provider_summary"):
        if selected:
            summary = st.columns(3)
            summary[0].metric(
                "当前厂商",
                selected.vendor_name or "未填写",
                border=True,
            )
            summary[1].metric(
                "默认模型",
                selected.model or "未填写",
                border=True,
            )
            summary[2].metric(
                "密钥状态",
                "已安全保存" if selected.api_key else "尚未设置",
                border=True,
            )
        else:
            st.caption("正在创建新的厂商配置。保存后会在上方显示配置摘要。")

    with st.expander(
        "编辑厂商配置",
        expanded=selected is None,
        icon=":material/tune:",
    ):
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
        st.caption(f"接口地址：{selected.base_url}")
        action_col_1, action_col_2, action_col_3 = st.columns(3)
        if action_col_1.button(
            "设为当前配置",
            icon=":material/check_circle:",
            width="stretch",
        ):
            PROVIDERS.set_active(selected.id)
            st.session_state["_provider_settings_notice"] = "当前配置已切换"
            _rerun()
        if action_col_2.button(
            "测试连接",
            icon=":material/sync:",
            width="stretch",
        ):
            try:
                with st.spinner("正在测试 AI 连接..."):
                    message = OpenAICompatibleClient(selected).test_connection()
                st.success(message)
            except AIRequestError as exc:
                st.error(str(exc))
            except Exception as exc:
                _show_error(exc)
        if action_col_3.button(
            "清除密钥",
            icon=":material/key_off:",
            width="stretch",
        ):
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
    append_name = ""
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
        append_name = next(
            item["name"] for item in existing if item["id"] == selected_existing
        )

    with st.form(f"import_{material_type}"):
        if append_mode:
            name = append_name
        else:
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
        with st.expander(
            "补充材料信息（可选）",
            icon=":material/info:",
        ):
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


def _render_material_browser_path(
    material_type: str,
    material: dict[str, Any] | None,
    document: dict[str, Any] | None,
) -> None:
    material_type_label = MATERIAL_LABELS[material_type]
    material_key = f"material_browser_material_{material_type}"
    document_key = f"material_browser_document_{material_type}"
    with st.container(
        horizontal=True,
        vertical_alignment="center",
        key=f"material_browser_path_{material_type}",
    ):
        st.markdown("材料中心 /")
        if st.button(
            material_type_label,
            type="tertiary",
            key=f"material_browser_path_root_{material_type}",
        ):
            st.session_state[material_key] = None
            st.session_state[document_key] = None
            st.rerun()
        if material is not None:
            st.markdown("/")
            if document is None:
                st.markdown(f"**{material['name']}**")
            else:
                if st.button(
                    material["name"],
                    type="tertiary",
                    key=f"material_browser_path_material_{material_type}",
                ):
                    st.session_state[document_key] = None
                    st.rerun()
                st.markdown("/")
                st.markdown(f"**{document['file_name']}**")


def _render_material_browser(material_type: str) -> None:
    material_type_label = MATERIAL_LABELS[material_type]
    material_key = f"material_browser_material_{material_type}"
    document_key = f"material_browser_document_{material_type}"
    materials = REPO.list_material_sets(material_type=material_type)
    material_by_id = {str(item["id"]): item for item in materials}
    selected_material_id = st.session_state.get(material_key)
    if selected_material_id not in material_by_id:
        selected_material_id = None
        st.session_state[material_key] = None
        st.session_state[document_key] = None
    selected_material = (
        material_by_id[str(selected_material_id)]
        if selected_material_id
        else None
    )
    documents = (
        REPO.list_documents(str(selected_material["id"]))
        if selected_material
        else []
    )
    document_by_id = {str(item["id"]): item for item in documents}
    selected_document_id = st.session_state.get(document_key)
    if selected_document_id not in document_by_id:
        selected_document_id = None
        st.session_state[document_key] = None
    selected_document = (
        document_by_id[str(selected_document_id)]
        if selected_document_id
        else None
    )
    _render_material_browser_path(
        material_type,
        selected_material,
        selected_document,
    )

    if selected_material is None:
        if not materials:
            st.info(f"尚未导入{material_type_label}，请切换到“导入”栏目。")
            return
        with st.container(key="material_browser_files"):
            with st.grid("auto", min_column_width=220, gap="small"):
                for material in materials:
                    with st.container(
                        border=True,
                        key=f"material_catalog_card_{material['id']}",
                    ):
                        st.markdown(f"**{material['name']}**")
                        st.caption(
                            f"第 {material['version']} 版 · "
                            f"{len(REPO.list_documents(material['id']))} 个切片"
                        )
                        if st.button(
                            "查看",
                            key=f"material_open_{material['id']}",
                            width="stretch",
                        ):
                            st.session_state[material_key] = material["id"]
                            st.session_state[document_key] = None
                            st.rerun()
        return

    if selected_document is None:
        if not documents:
            st.info("该材料还没有切片。")
            return
        with st.container(key="material_browser_files"):
            with st.grid("auto", min_column_width=200, gap="small"):
                for document in documents:
                    with st.container(
                        border=True,
                        key=f"material_catalog_card_{document['id']}",
                    ):
                        st.markdown(f"**{document['file_name']}**")
                        st.caption(
                            f"第 {document['order_index']} 个切片 · "
                            f"{document['page_count']} 页 · "
                            f"{document['extraction_status']}"
                        )
                        if st.button(
                            "查看切片",
                            key=f"material_document_open_{document['id']}",
                            icon=":material/description:",
                            width="stretch",
                        ):
                            st.session_state[document_key] = document["id"]
                            st.rerun()
        return

    pages = REPO.list_pages(selected_document["id"])
    sections = REPO.list_sections(selected_document["id"])
    units = REPO.list_units(document_id=selected_document["id"])
    warning_pages = [page for page in pages if page["status"] != "ok"]
    with st.container(
        horizontal=True,
        wrap=True,
        gap="xsmall",
        key="material_slice_badges",
    ):
        st.badge(
            "已结构化" if units else "未结构化",
            icon=(
                ":material/check_circle:"
                if units
                else ":material/pending:"
            ),
            color="green" if units else "orange",
        )
        st.badge(
            f"页数 {len(pages) or selected_document['page_count']}",
            icon=":material/description:",
            color="blue",
        )
        st.badge(
            f"章节块 {len(sections)}",
            icon=":material/format_list_bulleted:",
            color="gray",
        )
        st.badge(
            f"提取状态 {selected_document['extraction_status']}",
            icon=":material/sync:",
            color="violet",
        )
        st.badge(
            f"OCR 异常页 {len(warning_pages)}",
            icon=":material/warning:",
            color="red" if warning_pages else "gray",
        )

    directory, content = st.columns([1, 3], gap="large")
    with directory:
        with st.container(key="material_directory"):
            st.markdown("**目录**")
            if units:
                for unit in units:
                    st.markdown(f"**{unit['title']}**")
                    st.badge(
                        STATUS_LABELS.get(unit["status"], unit["status"]),
                        color={
                            UnitStatus.VERIFIED.value: "green",
                            UnitStatus.NEEDS_REVIEW.value: "orange",
                            UnitStatus.DRAFT.value: "gray",
                        }.get(unit["status"], "blue"),
                    )
            else:
                st.caption("该切片尚未结构化。")
                if st.button(
                    "前往编辑并开始结构化",
                    type="primary",
                    icon=":material/edit:",
                    width="stretch",
                    key=f"material_go_edit_{selected_document['id']}",
                ):
                    st.session_state["_pending_material_section"] = {
                        "item": (
                            "textbook"
                            if material_type == MaterialType.TEXTBOOK.value
                            else "standard"
                        ),
                        "section": "编辑",
                    }
                    st.session_state["_pending_material_library_id"] = (
                        selected_material["id"]
                    )
                    st.session_state["_pending_material_library_document_id"] = (
                        selected_document["id"]
                    )
                    st.rerun()
    with content:
        if units:
            with st.container(key="material_unit_blocks"):
                for unit in units:
                    with st.container(
                        key=f"material_unit_block_{unit['id']}",
                    ):
                        st.markdown(f"### {unit['title']}")
                        st.divider()
                        st.markdown(unit["content"])
        else:
            st.info("完成结构化后，这里将按目录显示各个单元的内容块。")


def render_material_library(
    material_type: str,
    *,
    mode: str,
) -> None:
    if mode not in {"browse", "edit"}:
        raise ValueError("未知材料中心材料栏目")
    material_type_label = MATERIAL_LABELS[material_type]
    st.subheader(f"{material_type_label}材料库")
    materials = [
        item
        for item in REPO.list_material_sets(material_type=material_type)
    ]
    if not materials:
        st.info(f"尚未导入{material_type_label}材料，请切换到“导入”栏目。")
        return
    material_query = st.text_input(
        f"搜索{material_type_label}材料",
        type="search",
        placeholder="输入名称、学科、年级、版本或备注",
        key=f"material_library_search_{material_type}_{mode}",
    )
    if material_query.strip():
        needle = material_query.casefold().strip()
        materials = [
            item
            for item in materials
            if any(
                needle in str(item.get(key) or "").casefold()
                for key in (
                    "name",
                    "subject",
                    "stage",
                    "grade",
                    "semester",
                    "edition",
                    "notes",
                )
            )
        ]
        if not materials:
            st.info("没有匹配当前搜索条件的材料。")
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
        if mode == "edit":
            with st.form("material_metadata"):
                columns = st.columns(3)
                subject = columns[0].text_input(
                    "学科",
                    value=material["subject"] or "",
                )
                stage = columns[1].text_input("学段", value=material["stage"] or "")
                grade = columns[2].text_input("年级", value=material["grade"] or "")
                columns_2 = st.columns(3)
                semester = columns_2[0].text_input(
                    "学期",
                    value=material["semester"] or "",
                )
                edition = columns_2[1].text_input(
                    "教材版本",
                    value=material["edition"] or "",
                )
                notes = columns_2[2].text_input(
                    "备注",
                    value=material["notes"] or "",
                )
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
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "字段": label,
                            "内容": material.get(key) or "未填写",
                        }
                        for key, label in (
                            ("subject", "学科"),
                            ("stage", "学段"),
                            ("grade", "年级"),
                            ("semester", "学期"),
                            ("edition", "教材版本"),
                            ("notes", "备注"),
                        )
                    ]
                ),
                hide_index=True,
                width="stretch",
                alt=f"{material_type_label}材料元数据",
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
    detail_query = st.text_input(
        "搜索章节或单元",
        type="search",
        placeholder="输入章节标题、切分方式或单元标题",
        key=f"material_detail_search_{material_type}_{mode}_{document_id}",
    )
    detail_needle = detail_query.casefold().strip()

    action_columns = st.columns(3)
    if mode == "edit" and action_columns[0].button(
        "提取/继续切分",
        width="stretch",
    ):
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
    if mode == "edit" and action_columns[1].button(
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
    if mode == "edit" and action_columns[2].button(
        "刷新状态",
        width="stretch",
    ):
        _rerun()
    if mode == "edit" and not sections:
        st.info("请先完成文字提取和章节切分，再调用 AI。")
    elif mode == "edit" and profile is None:
        st.warning("尚未设置当前 AI 厂商，无法开始结构化。")

    with st.expander(f"查看章节切分（{len(sections)}）"):
        if sections:
            visible_sections = [
                item
                for item in sections
                if not detail_needle
                or any(
                    detail_needle in str(item.get(key) or "").casefold()
                    for key in ("title", "path", "split_method", "id")
                )
            ]
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
                        for item in visible_sections
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
            if mode == "edit":
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
                col_text.text_area(
                    "识别文本",
                    value=page["final_text"],
                    height=360,
                    disabled=True,
                    key=f"page_text_read_{document_id}_{page_number}",
                )
        else:
            st.success("没有低置信度、空白或识别失败页。")

    if mode == "browse":
        st.subheader("结构化单元")
        visible_units = [
            unit
            for unit in units
            if not detail_needle
            or any(
                detail_needle in str(unit.get(key) or "").casefold()
                for key in (
                    "title",
                    "unit_type",
                    "content",
                    "section_title",
                )
            )
        ]
        if visible_units:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "类型": unit["unit_type"],
                            "标题": unit["title"],
                            "章节": unit["section_title"],
                            "状态": STATUS_LABELS.get(
                                unit["status"],
                                unit["status"],
                            ),
                            "版本": unit["current_version"],
                        }
                        for unit in visible_units
                    ]
                ),
                hide_index=True,
                width="stretch",
                alt=f"{material_type_label}结构化单元浏览",
            )
            selected_unit_id = st.selectbox(
                "查看单元内容",
                [unit["id"] for unit in visible_units],
                format_func=lambda value: next(
                    (
                        unit["title"]
                        for unit in visible_units
                        if unit["id"] == value
                    ),
                    value,
                ),
                key=f"material_browse_unit_{document_id}",
            )
            selected_unit = next(
                unit for unit in visible_units if unit["id"] == selected_unit_id
            )
            st.markdown(selected_unit["content"])
        else:
            st.info("没有匹配当前搜索条件的结构化单元。")
        return

    st.divider()
    st.subheader("逐单元审核")
    if not units:
        st.info("尚未生成结构化单元。")
        return
    searchable_units = [
        unit
        for unit in units
        if not detail_needle
        or any(
            detail_needle in str(unit.get(key) or "").casefold()
            for key in ("title", "unit_type", "content", "section_title")
        )
    ]
    if not searchable_units:
        st.info("没有匹配当前搜索条件的结构化单元。")
        return
    filter_columns = st.columns(2)
    status_filter = filter_columns[0].multiselect(
        "审核状态",
        [status.value for status in UnitStatus],
        default=[status.value for status in UnitStatus],
        format_func=lambda value: STATUS_LABELS.get(value, value),
    )
    section_options = list(
        dict.fromkeys(unit["section_id"] for unit in searchable_units)
    )
    section_filter = filter_columns[1].multiselect(
        "章节",
        section_options,
        default=section_options,
        format_func=lambda value: next(
            (
                unit["section_title"]
                for unit in searchable_units
                if unit["section_id"] == value
            ),
            value,
        ),
    )
    visible = [
        unit
        for unit in searchable_units
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


def _material_center_nav_label(item: dict[str, Any], *, active: bool) -> str:
    title = f"**{item['label']}**" if active else item["label"]
    return f"![]({'app/static/icons/' + item['icon']}) {title}"


def _render_material_center_top_nav() -> str:
    active_key = st.session_state.get("material_center_item")
    if active_key not in MATERIAL_CENTER_ITEM_BY_KEY:
        active_key = MATERIAL_CENTER_ITEMS[0]["key"]
        st.session_state["material_center_item"] = active_key
    with st.container(
        horizontal=True,
        wrap=False,
        gap="xsmall",
        key="material_center_nav",
    ):
        for item in MATERIAL_CENTER_ITEMS:
            if st.button(
                _material_center_nav_label(
                    item,
                    active=item["key"] == active_key,
                ),
                key=f"material_nav_{item['key']}",
                type="secondary",
                width="content",
            ):
                st.session_state["material_center_item"] = item["key"]
                st.rerun()
    st.html(
        f"""
        <style>
        .st-key-material_nav_{active_key} button {{
            background: #EDF2FF !important;
            border-color: #D6E1FF !important;
            color: #1F46C5 !important;
            font-weight: 700;
            box-shadow: inset 0 -2px 0 #2F5BEA;
        }}
        .st-key-material_nav_{active_key} button p {{
            font-weight: 700;
        }}
        </style>
        """
    )
    return str(active_key)


def _render_material_center_section_nav(item: dict[str, Any]) -> str:
    options = ["浏览", "编辑", "导入"]
    if item["key"] == "answers":
        options.append("异常检测")
    control_key = f"material_section_{item['key']}"
    pending = st.session_state.pop("_pending_material_section", None)
    if (
        isinstance(pending, dict)
        and pending.get("item") == item["key"]
        and pending.get("section") in options
    ):
        st.session_state[control_key] = pending["section"]
    if st.session_state.get(control_key) not in options:
        st.session_state[control_key] = "浏览"
    section = st.segmented_control(
        "材料操作",
        options,
        required=True,
        label_visibility="collapsed",
        width="stretch",
        wrap=False,
        key=control_key,
    )
    return str(section or "浏览")


def render_material_center() -> None:
    active_key = _render_material_center_top_nav()
    item = MATERIAL_CENTER_ITEM_BY_KEY[active_key]
    section = _render_material_center_section_nav(item)
    if item["kind"] == "material":
        if section == "导入":
            render_import_tab(item["data_type"])
        elif section == "浏览":
            _render_material_browser(item["data_type"])
        else:
            render_material_library(
                item["data_type"],
                mode="edit",
            )
        return

    dataset_id = render_dataset_context(LEARNING_DATA)
    if dataset_id is None:
        return
    if section == "异常检测":
        render_answer_anomalies(LEARNING_DATA, dataset_id)
        return
    mode = {
        "浏览": "browse",
        "编辑": "edit",
        "导入": "import",
    }[section]
    render_data_type_page(
        LEARNING_DATA,
        dataset_id,
        item["data_type"],
        mode=mode,
    )


def main() -> None:
    apply_app_style()
    page = render_sidebar_navigation()
    render_page_header(page)
    if page == "AI 参数设置":
        render_ai_settings()
    elif page == "材料中心":
        render_material_center()
    elif page == "学习内容梳理":
        render_outline_page(OUTLINES, _active_profile())
    elif page == "学生分析":
        render_student_analysis_page(
            ANALYTICS,
            LEARNING_DATA,
            DIAGNOSES,
            _active_profile(),
        )
    elif page == "班级分析":
        render_class_analysis_page(ANALYTICS, LEARNING_DATA)
    elif page == "教学建议":
        render_teaching_recommendations_page(
            TEACHING_RECOMMENDATIONS,
            LEARNING_DATA,
            _active_profile(),
        )


main()
