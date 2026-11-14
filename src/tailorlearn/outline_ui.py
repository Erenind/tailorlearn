from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from tailorlearn.models import OutlineItemStatus, ProviderProfile
from tailorlearn.outline import (
    ITEM_TYPE_LABELS,
    OUTLINE_FIELD_DEFINITIONS,
    RELATION_TYPE_LABELS,
    SCOPE_MODE_LABELS,
    SOURCE_KIND_LABELS,
    STATUS_LABELS,
    OutlineService,
)

OUTLINE_UI_VERSION = "2026-11-11-outline-history-state-fix"


def _rerun() -> None:
    st.rerun()


def _notify(message: str) -> None:
    st.session_state["_outline_notice"] = message


def _show_notice() -> None:
    message = st.session_state.pop("_outline_notice", None)
    if message:
        st.success(message)


def _section_label(row: dict[str, Any], show_theme: bool = False) -> str:
    prefix = f"[{row.get('theme_code') or '未分配'}] " if show_theme else ""
    chapter = row.get("chapter") or row.get("path") or "未分章节"
    return (
        f"{prefix}{row['file_name']} · {chapter} / {row['title']}"
        f"（第 {row['page_start']}–{row['page_end']} 页）"
    )


def _render_selectors(
    service: OutlineService,
    dataset_id: str,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    themes = service.list_themes(dataset_id)
    theme_codes = [theme["theme_code"] for theme in themes]
    theme_key = f"outline_theme_selector_{dataset_id}"
    if theme_key not in st.session_state:
        st.session_state[theme_key] = theme_codes
    selected_themes = st.multiselect(
        "主题范围",
        theme_codes,
        key=theme_key,
        format_func=lambda code: next(
            (
                f"{theme['theme_code']} {theme['theme_name']}"
                for theme in themes
                if theme["theme_code"] == code
            ),
            code,
        ),
    )
    standard_materials = service.materials("curriculum_standard")
    textbook_materials = service.materials("textbook")
    columns = st.columns(2)
    standard_material: dict[str, Any] | None = None
    standard_document: dict[str, Any] | None = None
    textbook_material: dict[str, Any] | None = None
    textbook_documents: list[dict[str, Any]] = []
    if standard_materials:
        standard_ids = [item["id"] for item in standard_materials]
        standard_key = "outline_standard_material_selector"
        if st.session_state.get(standard_key) not in standard_ids:
            st.session_state[standard_key] = standard_ids[0]
        standard_id = columns[0].selectbox(
            "课程标准材料",
            standard_ids,
            key=standard_key,
            format_func=lambda value: next(
                item["name"] for item in standard_materials if item["id"] == value
            ),
        )
        standard_material = next(
            item for item in standard_materials if item["id"] == standard_id
        )
        documents = service.documents(standard_id)
        if documents:
            document_ids = [item["id"] for item in documents]
            document_key = f"outline_standard_document_{standard_id}"
            if st.session_state.get(document_key) not in document_ids:
                st.session_state[document_key] = document_ids[0]
            document_id = columns[0].selectbox(
                "课程标准切片",
                document_ids,
                key=document_key,
                format_func=lambda value: next(
                    item["file_name"] for item in documents if item["id"] == value
                ),
            )
            standard_document = next(
                item for item in documents if item["id"] == document_id
            )
    else:
        columns[0].warning("尚未导入课程标准材料。")
    if textbook_materials:
        textbook_ids = [item["id"] for item in textbook_materials]
        textbook_key = "outline_textbook_material_selector"
        if st.session_state.get(textbook_key) not in textbook_ids:
            st.session_state[textbook_key] = textbook_ids[0]
        textbook_id = columns[1].selectbox(
            "教材材料",
            textbook_ids,
            key=textbook_key,
            format_func=lambda value: next(
                item["name"] for item in textbook_materials if item["id"] == value
            ),
        )
        textbook_material = next(
            item for item in textbook_materials if item["id"] == textbook_id
        )
        textbooks = service.documents(textbook_id)
        if textbooks:
            textbook_document_ids = [item["id"] for item in textbooks]
            document_key = f"outline_textbook_documents_{textbook_id}"
            if document_key not in st.session_state:
                st.session_state[document_key] = textbook_document_ids
            selected_ids = columns[1].multiselect(
                "教材切片",
                textbook_document_ids,
                key=document_key,
                format_func=lambda value: next(
                    f"{item['order_index']}. {item['file_name']}"
                    for item in textbooks
                    if item["id"] == value
                ),
            )
            textbook_documents = [
                item for item in textbooks if item["id"] in selected_ids
            ]
    else:
        columns[1].warning("尚未导入教材材料。")
    return (
        themes,
        selected_themes,
        standard_material,
        standard_document,
        [textbook_material] if textbook_material else [],
        textbook_documents,
    )


def _render_scope_editor(
    service: OutlineService,
    dataset_id: str,
    theme_codes: list[str],
    standard_material: dict[str, Any] | None,
    standard_document: dict[str, Any] | None,
    textbook_material: dict[str, Any] | None,
    textbook_documents: list[dict[str, Any]],
    themes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not theme_codes:
        st.warning("请至少选择一个主题。")
        return None
    if standard_material is None or standard_document is None:
        st.warning("请先导入并选择课程标准材料。")
        return None
    if textbook_material is None or not textbook_documents:
        st.warning("请先导入并选择教材材料。")
        return None
    document_ids = [item["id"] for item in textbook_documents]
    key = (
        f"outline_sections_{dataset_id}_"
        f"{'-'.join(theme_codes)}_{textbook_material['id']}_"
        f"{'-'.join(document_ids)}"
    )
    try:
        suggestions = service.suggest_sections(
            dataset_id=dataset_id,
            theme_codes=theme_codes,
            textbook_material_id=textbook_material["id"],
            textbook_document_ids=document_ids,
        )
        options = service.section_options(
            textbook_material["id"],
            document_ids,
        )
    except Exception as exc:
        st.error(str(exc))
        return None
    option_by_id = {row["section_id"]: row for row in options}
    suggested_ids = [row["section_id"] for row in suggestions]
    if key not in st.session_state:
        st.session_state[key] = suggested_ids
    else:
        st.session_state[key] = [
            section_id
            for section_id in st.session_state[key]
            if section_id in option_by_id
        ]
    selected_ids = st.multiselect(
        "纳入本次梳理的教材章节",
        list(option_by_id),
        key=key,
        format_func=lambda section_id: _section_label(
            option_by_id[section_id]
        ),
        help=(
            "默认使用按主题自动匹配的小节；可增删。"
            "同一章内的小节按教材顺序进入范围。"
        ),
    )
    notice_columns = st.columns([1, 3])
    if notice_columns[0].button("重新自动匹配", width="stretch"):
        st.session_state[key] = suggested_ids
        st.rerun()
    notice_columns[1].caption(
        f"自动匹配 {len(suggested_ids)} 个小节；当前纳入 {len(selected_ids)} 个。"
    )
    try:
        scope = service.build_scope(
            dataset_id=dataset_id,
            theme_codes=theme_codes,
            standard_material_id=standard_material["id"],
            standard_document_id=standard_document["id"],
            textbook_material_id=textbook_material["id"],
            textbook_document_ids=document_ids,
            section_ids=selected_ids or None,
        )
    except Exception as exc:
        st.error(str(exc))
        return None
    standard_entries = scope["standard"]["themes"]
    metrics = st.columns(5)
    metrics[0].metric("主题", len(scope["themes"]))
    metrics[1].metric(
        "观察指标",
        sum(len(theme["indicators"]) for theme in scope["themes"]),
    )
    standard_pages = sorted(
        {
            page
            for entry in standard_entries
            if entry["matched"]
            for page in range(entry["page_start"], entry["page_end"] + 1)
        }
    )
    metrics[2].metric(
        "课标页码",
        (
            f"{standard_pages[0]}–{standard_pages[-1]}"
            if standard_pages
            else "待确认"
        ),
    )
    metrics[3].metric("教材章节", len(scope["textbook"]["sections"]))
    metrics[4].metric("未纳入章节", len(scope["textbook"]["excluded"]))
    with st.expander("范围边界说明（可编辑）", expanded=True):
        statement = st.text_area(
            "已纳入范围",
            value=scope["boundary"]["statement"],
            height=110,
            key=f"outline_statement_{dataset_id}_{'-'.join(theme_codes)}",
        )
        if scope["boundary"]["excluded_note"]:
            st.caption(scope["boundary"]["excluded_note"])
        scope["boundary"]["statement"] = statement.strip()
    for warning in scope["boundary"]["warnings"]:
        st.warning(warning)
    with st.expander("查看范围内教材小节与未纳入章节"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "主题": row["theme_code"],
                        "切片": row["file_name"],
                        "章节": row["chapter"],
                        "小节": row["title"],
                        "页码": f"{row['page_start']}–{row['page_end']}",
                        "匹配度": row["match_score"],
                    }
                    for row in scope["textbook"]["sections"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        if scope["textbook"]["excluded"]:
            st.markdown("**未纳入本次梳理的章节**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "切片": row["file_name"],
                            "章节": row["chapter"],
                            "页码": f"{row['page_start']}–{row['page_end']}",
                            "小节数": row["section_count"],
                            "示例": "；".join(row["titles"]),
                        }
                        for row in scope["textbook"]["excluded"]
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
    return scope


def _render_generate(
    service: OutlineService,
    scope: dict[str, Any],
    profile: ProviderProfile | None,
) -> None:
    st.subheader("生成梳理")
    st.caption(
        "规则梳理依据课标条目、教材章节顺序和术语重合自动生成，"
        "不需要 API；AI 梳理会逐条回查出处，结果仍需教师确认。"
    )
    columns = st.columns([1, 1, 2])
    if columns[0].button("开始规则梳理", type="primary", width="stretch"):
        try:
            with st.spinner("正在生成学习内容与要求清单..."):
                result = service.generate(scope=scope, method="rule")
            _notify(
                f"已生成 {result['run']['item_count']} 条学习内容、"
                f"{result['run']['relation_count']} 条必要联系。"
            )
            st.session_state[f"outline_run_selector_{scope['dataset']['id']}"] = (
                result["run"]["id"]
            )
            _rerun()
        except Exception as exc:
            st.error(str(exc))
    ai_disabled = profile is None
    if columns[1].button(
        "AI 梳理",
        width="stretch",
        disabled=ai_disabled,
        help=(
            "需要先在“AI 参数设置”保存并启用厂商配置。"
            if ai_disabled
            else "使用当前厂商依据所选范围起草，并逐条核对出处。"
        ),
    ):
        if profile is None:
            st.error("请先配置当前 AI 厂商")
            return
        try:
            with st.spinner(f"正在使用 {profile.vendor_name} / {profile.model} 梳理..."):
                result = service.generate(
                    scope=scope,
                    method="ai",
                    profile=profile,
                )
            _notify(
                f"AI 已生成 {result['run']['item_count']} 条学习内容，"
                "请逐条核对课标与教材引文。"
            )
            st.session_state[f"outline_run_selector_{scope['dataset']['id']}"] = (
                result["run"]["id"]
            )
            _rerun()
        except Exception as exc:
            st.error(str(exc))
    if ai_disabled:
        columns[2].caption("当前未配置 AI 厂商，只能使用规则梳理。")


def _run_metrics(run: dict[str, Any], items: list[Any], relations: list[Any]) -> None:
    verified = sum(
        1
        for item in items
        for source in item.sources
        if source.verified
    )
    warnings = sum(len(item.review_warnings) for item in items)
    metrics = st.columns(5)
    metrics[0].metric("学习内容", len(items))
    metrics[1].metric("必要联系", len(relations))
    metrics[2].metric("通过核对出处", verified)
    metrics[3].metric("核对提示", warnings)
    metrics[4].metric(
        "生成方式",
        "AI 梳理" if run["method"] == "ai" else "规则梳理",
    )


def _render_items_tab(
    service: OutlineService,
    run: dict[str, Any],
    items: list[Any],
    relations: list[Any],
    profile: ProviderProfile | None,
) -> None:
    st.subheader("学习内容与要求清单")
    st.caption(
        "每条包含学习内容、课标要求、教材出处、对应指标和核对状态；"
        "未通过核对的条目保留并标注，不自动删除。"
    )
    if not items:
        st.info("本次梳理没有学习内容条目。")
        return
    themes = sorted({(item.theme_code, item.theme_name) for item in items})
    theme_by_code = dict(themes)
    filter_columns = st.columns(2)
    theme_filter = filter_columns[0].multiselect(
        "主题",
        list(theme_by_code),
        default=list(theme_by_code),
        format_func=lambda code: f"{code} {theme_by_code[code]}",
    )
    status_filter = filter_columns[1].multiselect(
        "核对状态",
        [status.value for status in OutlineItemStatus],
        default=[status.value for status in OutlineItemStatus],
        format_func=lambda value: STATUS_LABELS.get(value, value),
    )
    visible = [
        item
        for item in items
        if item.theme_code in theme_filter and item.status in status_filter
    ]
    if not visible:
        st.info("当前筛选条件下没有条目。")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "编号": item.item_key,
                    "主题": item.theme_name,
                    "类型": ITEM_TYPE_LABELS.get(item.item_type, item.item_type),
                    "学习内容": item.content,
                    "要求数": len(item.requirements),
                    "出处数": len(item.sources),
                    "指标": "、".join(item.indicator_codes),
                    "状态": STATUS_LABELS.get(item.status, item.status),
                    "核对提示": "；".join(item.review_warnings),
                }
                for item in visible
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    selected_keys = st.multiselect(
        "批量选择条目",
        [item.item_key for item in visible],
        format_func=lambda key: next(
            (item.content for item in visible if item.item_key == key),
            key,
        ),
        key=f"outline_batch_{run['id']}",
    )
    if selected_keys:
        selected_ids = [
            item.id for item in visible if item.item_key in selected_keys
        ]
        buttons = st.columns(3)
        if buttons[0].button("标记已确认", width="stretch"):
            count = service.set_items_status(
                selected_ids,
                OutlineItemStatus.VERIFIED.value,
            )
            _notify(f"已确认 {count} 条学习内容。")
            _rerun()
        if buttons[1].button("标记待复核", width="stretch"):
            count = service.set_items_status(
                selected_ids,
                OutlineItemStatus.NEEDS_REVIEW.value,
            )
            _notify(f"已标记 {count} 条为待复核。")
            _rerun()
        if buttons[2].button("标记草稿", width="stretch"):
            count = service.set_items_status(
                selected_ids,
                OutlineItemStatus.DRAFT.value,
            )
            _notify(f"已恢复 {count} 条为草稿。")
            _rerun()
    item_key = st.selectbox(
        "查看或编辑单个条目",
        [item.item_key for item in visible],
        format_func=lambda key: next(
            (f"{item.item_key} · {item.content}" for item in visible if item.item_key == key),
            key,
        ),
        key=f"outline_item_selector_{run['id']}",
    )
    item = next(item for item in visible if item.item_key == item_key)
    if item.review_warnings:
        st.warning("；".join(item.review_warnings))
    detail_columns = st.columns([3, 2])
    with detail_columns[0]:
        st.markdown("**对应课程标准要求**")
        if item.requirements:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "层级": requirement.level,
                            "要求原文": requirement.text,
                            "课标页": requirement.evidence.page
                            if requirement.evidence
                            else "",
                            "正文页": requirement.evidence.printed_page
                            if requirement.evidence
                            else "",
                            "核对": "通过" if requirement.verified else "待核对",
                            "说明": requirement.note,
                        }
                        for requirement in item.requirements
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("尚未附对应课标要求。")
    with detail_columns[1]:
        st.markdown("**学习要点**")
        if item.key_points:
            for point in item.key_points:
                st.markdown(f"- {point}")
        else:
            st.caption("尚未整理要点。")
    st.markdown("**材料出处**")
    if item.sources:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "类型": SOURCE_KIND_LABELS.get(source.kind, source.kind),
                        "材料": source.material_name,
                        "文件": source.file_name,
                        "章节": source.section_path,
                        "页码": source.page,
                        "原文": source.quote,
                        "核对": "通过" if source.verified else "待核对",
                        "说明": source.verification_note,
                    }
                    for source in item.sources
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("尚未附教材材料出处。")
    with st.expander("编辑该条目"):
        with st.form(f"outline_edit_{item.id}"):
            content = st.text_input("学习内容", value=item.content)
            key_points = st.text_area(
                "学习要点（每行一条）",
                value="\n".join(item.key_points),
                height=120,
            )
            indicator_codes = st.text_input(
                "对应指标编号（用、或逗号分隔）",
                value="、".join(item.indicator_codes),
            )
            item_type = st.selectbox(
                "内容类型",
                list(ITEM_TYPE_LABELS),
                index=list(ITEM_TYPE_LABELS).index(item.item_type)
                if item.item_type in ITEM_TYPE_LABELS
                else 0,
                format_func=lambda value: ITEM_TYPE_LABELS[value],
            )
            status = st.selectbox(
                "核对状态",
                [status.value for status in OutlineItemStatus],
                index=(
                    [status.value for status in OutlineItemStatus].index(item.status)
                    if item.status in [status.value for status in OutlineItemStatus]
                    else 0
                ),
                format_func=lambda value: STATUS_LABELS.get(value, value),
            )
            notes = st.text_area("备注", value=item.notes, height=80)
            submitted = st.form_submit_button("保存为新版本")
        if submitted:
            try:
                service.update_item(
                    item_id=item.id,
                    content=content,
                    key_points=key_points.splitlines(),
                    indicator_codes=[
                        code.strip()
                        for code in indicator_codes.replace("，", "、")
                        .replace(",", "、")
                        .split("、")
                        if code.strip()
                    ],
                    notes=notes,
                    theme_code=item.theme_code,
                    theme_name=item.theme_name,
                    item_type=item_type,
                    status=status,
                )
                _notify("学习内容已保存为新版本。")
                _rerun()
            except Exception as exc:
                st.error(str(exc))
    delete_columns = st.columns([1, 3])
    if delete_columns[0].button("软删除该条目", width="stretch"):
        try:
            service.delete_item(item.id)
            _notify("已软删除该条目，成果文件已同步。")
            _rerun()
        except Exception as exc:
            st.error(str(exc))
    delete_columns[1].caption("软删除只从当前清单移除，数据库版本记录仍保留。")
    _render_manual_item(service, run, item)
    related = [
        relation
        for relation in relations
        if item.item_key in {
            relation.source_item_key,
            relation.target_item_key,
        }
    ]
    with st.expander(f"该条目的联系（{len(related)}）"):
        if related:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "源": relation.source_item_key,
                            "目标": relation.target_item_key,
                            "类型": RELATION_TYPE_LABELS.get(
                                relation.relation_type,
                                relation.relation_type,
                            ),
                            "说明": relation.note,
                        }
                        for relation in related
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("暂无联系。")


def _render_manual_item(
    service: OutlineService,
    run: dict[str, Any],
    item: Any,
) -> None:
    scope = service.scope_of(run)
    sections = scope.get("textbook", {}).get("sections", [])
    with st.expander("人工添加学习内容"):
        st.caption("人工添加的条目默认为待复核，出处仍需回查。")
        with st.form(f"outline_manual_item_{run['id']}"):
            theme_options = {
                theme["theme_code"]: theme["theme_name"]
                for theme in scope.get("themes", [])
            }
            theme_code = st.selectbox(
                "主题",
                list(theme_options),
                format_func=lambda code: f"{code} {theme_options[code]}",
            )
            content = st.text_input("学习内容")
            key_points = st.text_area("学习要点（每行一条）", height=100)
            indicator_codes = st.text_input(
                "对应指标编号（用、分隔）",
                value=item.indicator_codes[0] if item.indicator_codes else "",
            )
            section_by_id = {
                row["section_id"]: row for row in sections
            }
            section_id = st.selectbox(
                "教材章节（可选）",
                ["__none__", *section_by_id],
                format_func=lambda value: (
                    "不指定"
                    if value == "__none__"
                    else _section_label(section_by_id[value])
                ),
            )
            page = st.number_input("教材页码（可选）", min_value=0, value=0)
            quote = st.text_area("教材原文引文（可选）", height=80)
            standard_page = st.number_input("课标页码（可选）", min_value=0, value=0)
            standard_quote = st.text_area("课标原文引文（可选）", height=80)
            notes = st.text_area("备注", height=80)
            submitted = st.form_submit_button("添加")
        if submitted:
            try:
                service.add_manual_item(
                    run_id=run["id"],
                    theme_code=theme_code,
                    content=content,
                    key_points=key_points.splitlines(),
                    indicator_codes=[
                        code.strip()
                        for code in indicator_codes.replace("，", "、")
                        .replace(",", "、")
                        .split("、")
                        if code.strip()
                    ],
                    notes=notes,
                    section_id=None
                    if section_id == "__none__"
                    else section_id,
                    page=int(page) or None,
                    quote=quote,
                    standard_page=int(standard_page) or None,
                    standard_quote=standard_quote,
                )
                _notify("已添加人工条目。")
                _rerun()
            except Exception as exc:
                st.error(str(exc))


def _render_relations_tab(
    service: OutlineService,
    run: dict[str, Any],
    items: list[Any],
    relations: list[Any],
) -> None:
    st.subheader("必要联系")
    st.caption(
        "联系用于说明先备、应用、并列和对比关系；"
        "每条都写明依据，规则推断的联系标为待复核。"
    )
    if relations:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "源": relation.source_item_key,
                        "目标": relation.target_item_key,
                        "类型": RELATION_TYPE_LABELS.get(
                            relation.relation_type,
                            relation.relation_type,
                        ),
                        "说明": relation.note,
                        "依据": relation.basis,
                        "生成": {
                            "rule": "规则",
                            "ai": "AI",
                            "manual": "人工",
                        }.get(relation.origin, relation.origin),
                        "状态": STATUS_LABELS.get(relation.status, relation.status),
                    }
                    for relation in relations
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("尚无必要联系。")
    if items:
        with st.expander("人工添加联系"):
            with st.form(f"outline_manual_relation_{run['id']}"):
                keys = [item.item_key for item in items]
                source_key = st.selectbox("起点", keys)
                target_key = st.selectbox("终点", keys)
                relation_type = st.selectbox(
                    "联系类型",
                    list(RELATION_TYPE_LABELS),
                    format_func=lambda value: RELATION_TYPE_LABELS[value],
                )
                note = st.text_area("联系说明", height=90)
                basis = st.text_area("依据", height=70)
                submitted = st.form_submit_button("添加联系")
            if submitted:
                try:
                    source = next(
                        item for item in items if item.item_key == source_key
                    )
                    target = next(
                        item for item in items if item.item_key == target_key
                    )
                    service.add_relation(
                        run_id=run["id"],
                        source_item_id=source.id,
                        target_item_id=target.id,
                        relation_type=relation_type,
                        note=note,
                        basis=basis,
                    )
                    _notify("已添加联系。")
                    _rerun()
                except Exception as exc:
                    st.error(str(exc))
    if relations:
        with st.expander("删除联系"):
            selected = st.selectbox(
                "选择联系",
                [relation.id for relation in relations],
                format_func=lambda value: next(
                    (
                        f"{relation.source_item_key} → "
                        f"{relation.target_item_key} "
                        f"（{RELATION_TYPE_LABELS.get(relation.relation_type, relation.relation_type)}）"
                        for relation in relations
                        if relation.id == value
                    ),
                    value,
                ),
            )
            if st.button("删除该联系"):
                try:
                    service.delete_relation(selected)
                    _notify("已删除联系。")
                    _rerun()
                except Exception as exc:
                    st.error(str(exc))


def _render_sources_tab(
    service: OutlineService,
    run: dict[str, Any],
    items: list[Any],
    relations: list[Any],
) -> None:
    st.subheader("材料出处与核对")
    st.caption(
        "出处包含课程标准与教材的页面、原文引文和核对结果；"
        "未通过的引文保留在清单中，需人工修订页面文本或条目。"
    )
    scope = service.scope_of(run)
    boundary_data = scope.get("boundary", {})
    with st.expander("范围边界", expanded=True):
        boundary = st.text_area(
            "范围边界说明",
            value=boundary_data.get("statement") or run["boundary_note"],
            height=130,
            key=f"outline_boundary_{run['id']}",
        )
        if st.button("保存范围边界说明"):
            try:
                service.update_run_boundary(run["id"], boundary)
                st.session_state.pop(f"outline_boundary_{run['id']}", None)
                _notify("范围边界说明已保存。")
                _rerun()
            except Exception as exc:
                st.error(str(exc))
        st.caption(boundary_data.get("excluded_note", ""))
    references = service.references(items, relations)
    metrics = st.columns(4)
    metrics[0].metric("出处条数", len(references))
    metrics[1].metric(
        "通过核对",
        sum(1 for row in references if row["verified"]),
    )
    metrics[2].metric(
        "待核对",
        sum(1 for row in references if not row["verified"]),
    )
    metrics[3].metric("教材章节", len(scope.get("textbook", {}).get("sections", [])))
    if references:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "出处编号": row["ref_id"],
                        "类型": row["kind"],
                        "材料": row["material"],
                        "文件": row["file"],
                        "章节": row["section"],
                        "页码": row["page"],
                        "正文页": row["printed_page"],
                        "原文": row["quote"],
                        "核对": "通过" if row["verified"] else "待核对",
                        "用于": "、".join(row["used_by"]),
                    }
                    for row in references
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    unverified = [row for row in references if not row["verified"]]
    if unverified:
        st.warning(f"{len(unverified)} 条出处引文待核对。")
        for row in unverified[:10]:
            st.markdown(
                f"- {row['ref_id']} · {row['file']} 第 {row['page']} 页："
                f"{row['quote'][:80]}"
            )
    with st.expander("查看范围内教材小节"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "主题": row["theme_code"],
                        "切片": row["file_name"],
                        "章节": row["chapter"],
                        "小节": row["title"],
                        "页码": f"{row['page_start']}–{row['page_end']}",
                    }
                    for row in scope.get("textbook", {}).get("sections", [])
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def _render_export_tab(
    service: OutlineService,
    run: dict[str, Any],
    items: list[Any],
    relations: list[Any],
) -> None:
    st.subheader("导出与字段说明")
    st.caption(
        "成果包同时写出 JSON/JSONL/CSV；CSV 使用 UTF-8 BOM，"
        "可用 Excel 直接打开。"
    )
    try:
        bundle = service.export_bundle(run["id"])
        items_csv = service.export_items_csv(run["id"])
        relations_csv = service.export_relations_csv(run["id"])
        references_csv = service.export_references_csv(run["id"])
        fields_csv = service.export_field_definitions_csv()
    except Exception as exc:
        st.error(str(exc))
        return
    columns = st.columns(2)
    columns[0].download_button(
        "下载完整成果包（JSON）",
        data=bundle,
        file_name=f"{run['id']}_outline.json",
        mime="application/json",
        width="stretch",
    )
    columns[1].download_button(
        "下载学习内容清单（CSV）",
        data=items_csv,
        file_name=f"{run['id']}_items.csv",
        mime="text/csv",
        width="stretch",
    )
    columns = st.columns(2)
    columns[0].download_button(
        "下载必要联系（CSV）",
        data=relations_csv,
        file_name=f"{run['id']}_relations.csv",
        mime="text/csv",
        width="stretch",
    )
    columns[1].download_button(
        "下载材料出处（CSV）",
        data=references_csv,
        file_name=f"{run['id']}_references.csv",
        mime="text/csv",
        width="stretch",
    )
    columns = st.columns(2)
    columns[0].download_button(
        "下载字段说明（CSV）",
        data=fields_csv,
        file_name="outline_field_definitions.csv",
        mime="text/csv",
        width="stretch",
    )
    artifact = service.outlines_dir / run["id"]
    if columns[1].button("重新写出成果文件", width="stretch"):
        try:
            service.write_artifacts(run["id"])
            _notify(f"成果文件已写出到 {artifact}")
            _rerun()
        except Exception as exc:
            st.error(str(exc))
    st.code(str(artifact), language=None)
    st.caption(
        "成果包含 scope.json、items.jsonl/csv、relations.jsonl/csv、"
        "references.json/csv、field_definitions.csv 和 README.md。"
    )
    with st.expander("字段说明", expanded=True):
        st.dataframe(
            pd.DataFrame(OUTLINE_FIELD_DEFINITIONS),
            hide_index=True,
            width="stretch",
        )
    runs = service.list_runs(run["dataset_id"])
    with st.expander("历史梳理记录"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "编号": row["id"],
                        "名称": row["name"],
                        "方式": "AI" if row["method"] == "ai" else "规则",
                        "条目": row["item_count"],
                        "联系": row["relation_count"],
                        "时间": row["created_at"],
                    }
                    for row in runs
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def _history_label(service: OutlineService, run: dict[str, Any]) -> str:
    scope = service.scope_of(run)
    mode = SCOPE_MODE_LABELS.get(
        str(scope.get("mode") or ""),
        str(scope.get("mode_label") or "旧格式"),
    )
    dataset = scope.get("dataset") or {}
    dataset_name = dataset.get("name") if mode == "主题模式" else "不涉及学情数据"
    return (
        f"{mode} · {dataset_name} · {run['created_at'][:16].replace('T', ' ')}"
        f" · {run['item_count']} 条"
    )


def _render_outline_settings(
    service: OutlineService,
    profile: ProviderProfile | None,
) -> tuple[dict[str, Any] | None, bool]:
    with st.container(border=True, key="outline_settings_panel"):
        st.subheader("梳理设置")
        mode_label = st.segmented_control(
            "范围模式",
            ["课本模式", "主题模式"],
            default="课本模式",
            key="outline_scope_mode",
            required=True,
        )
        mode = "textbook" if mode_label == "课本模式" else "theme"
        dataset_id: str | None = None
        theme_codes: list[str] = []
        textbook_material_ids: list[str] = []
        textbook_document_ids: list[str] = []

        datasets = service.list_datasets()
        if mode == "textbook":
            st.info(
                "课本模式不涉及学情数据，仅使用课本和课程标准。",
                icon=":material/info:",
            )
            materials = service.materials("textbook")
            if materials:
                material_ids = [item["id"] for item in materials]
                material_key = "outline_textbook_scope_material"
                if st.session_state.get(material_key) not in material_ids:
                    st.session_state[material_key] = material_ids[0]
                material_id = st.selectbox(
                    "课本",
                    material_ids,
                    key=material_key,
                    format_func=lambda value: next(
                        f"{item['name']} · 第 {item['version']} 版"
                        for item in materials
                        if item["id"] == value
                    ),
                )
                textbook_material_ids = [material_id]
                documents = service.documents(material_id)
                doc_ids = [document["id"] for document in documents]
                textbook_document_ids = st.multiselect(
                    "课本切片",
                    doc_ids,
                    key=f"outline_textbook_scope_docs_{material_id}",
                    format_func=lambda value: next(
                        document["file_name"]
                        for document in documents
                        if document["id"] == value
                    ),
                )
            else:
                st.warning("尚未导入课本。")
        else:
            if datasets:
                dataset_ids = [item["id"] for item in datasets]
                dataset_key = "outline_theme_scope_dataset"
                if st.session_state.get(dataset_key) not in dataset_ids:
                    st.session_state[dataset_key] = dataset_ids[0]
                dataset_id = st.selectbox(
                    "学情数据集",
                    dataset_ids,
                    key=dataset_key,
                    format_func=lambda value: next(
                        item["name"] for item in datasets if item["id"] == value
                    ),
                )
                themes = service.list_themes(dataset_id)
                theme_by_code = {
                    theme["theme_code"]: theme for theme in themes
                }
                theme_codes = st.multiselect(
                    "主题",
                    list(theme_by_code),
                    key=f"outline_theme_scope_themes_{dataset_id}",
                    format_func=lambda value: (
                        f"{value} {theme_by_code[value]['theme_name']}"
                    ),
                )
            else:
                st.warning("主题模式需要先导入学情数据和课程要求。")
            materials = service.materials("textbook")
            textbook_material_ids = st.multiselect(
                "课本",
                [item["id"] for item in materials],
                key="outline_theme_scope_textbooks",
                format_func=lambda value: next(
                    f"{item['name']} · 第 {item['version']} 版"
                    for item in materials
                    if item["id"] == value
                ),
            )

        st.markdown("**课程标准版本**")
        standard_materials = service.materials("curriculum_standard")
        standard_material_id: str | None = None
        standard_document_id: str | None = None
        if standard_materials:
            standard_ids = [item["id"] for item in standard_materials]
            standard_key = "outline_standard_scope_material"
            if st.session_state.get(standard_key) not in standard_ids:
                st.session_state[standard_key] = standard_ids[0]
            standard_material_id = st.selectbox(
                "课程标准材料",
                standard_ids,
                key=standard_key,
                format_func=lambda value: next(
                    f"{item['name']} · 第 {item['version']} 版"
                    for item in standard_materials
                    if item["id"] == value
                ),
            )
            standard_documents = service.documents(standard_material_id)
            standard_document_ids = [
                document["id"] for document in standard_documents
            ]
            if standard_document_ids:
                doc_key = f"outline_standard_scope_doc_{standard_material_id}"
                if st.session_state.get(doc_key) not in standard_document_ids:
                    st.session_state[doc_key] = standard_document_ids[0]
                standard_document_id = st.selectbox(
                    "课程标准切片",
                    standard_document_ids,
                    key=doc_key,
                    format_func=lambda value: next(
                        document["file_name"]
                        for document in standard_documents
                        if document["id"] == value
                    ),
                )
        else:
            st.warning("尚未导入课程标准。")

        with st.expander(
            "具体要求（结构化生成阶段）",
            expanded=False,
        ):
            specific_requirements = st.text_area(
                "具体要求",
                value=st.session_state.get(
                    "outline_specific_requirements", ""
                ),
                height=150,
                key="outline_specific_requirements",
                placeholder=(
                    "可留空。例如：重点整理先备关系，排除未教学内容；"
                    "学习内容按教材顺序编号。"
                ),
            )
        can_start = bool(
            profile
            and standard_material_id
            and standard_document_id
            and textbook_material_ids
            and (
                mode == "textbook"
                and textbook_document_ids
                or mode == "theme"
                and dataset_id
                and theme_codes
            )
        )
        start = st.button(
            "开始梳理",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
            disabled=not can_start,
            key="outline_two_stage_start",
        )
        if profile is None:
            st.warning("请先在“AI 参数设置”中启用厂商配置。")
    if not start:
        return None, False
    assert (
        profile is not None
        and standard_material_id is not None
        and standard_document_id is not None
    )
    return (
        {
            "mode": mode,
            "dataset_id": dataset_id,
            "theme_codes": theme_codes,
            "textbook_material_ids": textbook_material_ids,
            "textbook_document_ids": textbook_document_ids,
            "standard_material_id": standard_material_id,
            "standard_document_id": standard_document_id,
            "specific_requirements": specific_requirements.strip(),
        },
        True,
    )


def _directory_names(
    scope: dict[str, Any],
    selection: dict[str, Any],
) -> str:
    standard_by_id = {
        item["section_id"]: item["title"]
        for item in scope["standard"]["directory"]
    }
    textbook_by_id = {
        item["section_id"]: (
            f"{item['file_name']} · {item['path']}"
        )
        for item in scope["textbook"]["directory"]
    }
    names = [
        *[
            standard_by_id[item]
            for item in selection["standard_section_ids"]
            if item in standard_by_id
        ],
        *[
            textbook_by_id[item]
            for item in selection["textbook_section_ids"]
            if item in textbook_by_id
        ],
    ]
    return "；".join(names) if names else "无"


def _queue_history_run(run_id: str) -> None:
    st.session_state["_outline_pending_history_run"] = run_id


def _apply_pending_history_run() -> None:
    run_id = st.session_state.pop(
        "_outline_pending_history_run",
        None,
    )
    if run_id:
        st.session_state["outline_history_run"] = run_id


def _run_two_stage_generation(
    service: OutlineService,
    settings: dict[str, Any],
    profile: ProviderProfile,
) -> dict[str, Any] | None:
    process = st.container(border=True, key="outline_live_process")
    process.markdown("**处理过程**")
    progress = process.progress(0)
    try:
        process.markdown(
            "**01 范围准备与目录提交** · 正在构建范围并准备目录"
        )
        progress.progress(12)
        scope = service.prepare_two_stage_scope(
            mode=settings["mode"],
            dataset_id=settings["dataset_id"],
            theme_codes=settings["theme_codes"],
            standard_material_id=settings["standard_material_id"],
            standard_document_id=settings["standard_document_id"],
            textbook_material_ids=settings["textbook_material_ids"],
            textbook_document_ids=settings["textbook_document_ids"],
        )
        process.markdown(
            f"已提交范围：{scope['boundary']['statement']}"
        )
        process.markdown(
            f"课程标准目录 {len(scope['standard']['directory'])} 项；"
            f"课本目录 {len(scope['textbook']['directory'])} 项。"
        )
        progress.progress(24)
        process.markdown(
            "**02 目录检索阶段** · 正在匹配需要读取的课程标准与课本目录"
        )
        selection = service.select_materials(scope=scope, profile=profile)
        st.session_state["_outline_pending_stage"] = {
            "scope": scope,
            "selection": selection,
            "specific_requirements": settings["specific_requirements"],
        }
        process.markdown(
            "目录检索完成：需要读取"
            f"课程标准目录 {len(selection['standard_section_ids'])} 项、"
            f"课本目录 {len(selection['textbook_section_ids'])} 项。"
        )
        progress.progress(48)
        process.markdown(
            "**03 内容提取阶段** · 正在从本地读取："
            f"{_directory_names(scope, selection)}"
        )
        progress.progress(62)
        process.markdown(
            "**04 结构化生成阶段** · 正在生成学习内容、要求、来源与联系"
        )
        result = service.generate_from_selection(
            scope=scope,
            selection=selection,
            specific_requirements=settings["specific_requirements"],
            profile=profile,
        )
        progress.progress(90)
        process.markdown(
            "**05 结果校验与保存** · 正在核对来源并写入历史清单"
        )
        st.session_state.pop("_outline_pending_stage", None)
        _queue_history_run(result["run"]["id"])
        progress.progress(100)
        process.success(
            f"梳理完成：生成 {len(result['items'])} 条学习内容、"
            f"{len(result['relations'])} 条联系。"
        )
        return result
    except Exception as exc:
        process.markdown("**处理中止**")
        process.error(str(exc))
        return None


def _render_requirement_sources(item: Any) -> None:
    rows = []
    for requirement in item.requirements:
        evidence = requirement.evidence
        rows.append(
            {
                "要求": requirement.text,
                "层级": requirement.level,
                "课标页": evidence.page if evidence else "",
                "正文页": evidence.printed_page if evidence else "",
                "来源原文": evidence.quote if evidence else "",
                "核对": "通过" if requirement.verified else "待核对",
            }
        )
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            alt=f"{item.item_key}对应要求与来源",
        )
    else:
        st.caption("未附对应要求。")


def _render_item_source_content(item: Any) -> None:
    if not item.sources:
        st.caption("未附学习内容来源。")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "材料": source.material_name,
                    "文件": source.file_name,
                    "章节": source.section_path or source.section_id,
                    "页码": source.page,
                    "原文": source.quote,
                    "核对": "通过" if source.verified else "待核对",
                }
                for source in item.sources
            ]
        ),
        hide_index=True,
        width="stretch",
        alt=f"{item.item_key}学习内容来源",
    )


def _render_item_relations(
    item: Any,
    relations: list[Any],
) -> None:
    related = [
        relation
        for relation in relations
        if item.item_key
        in {relation.source_item_key, relation.target_item_key}
    ]
    if not related:
        st.caption("暂无与其他学习内容的联系。")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "起点": relation.source_item_key,
                    "终点": relation.target_item_key,
                    "类型": RELATION_TYPE_LABELS.get(
                        relation.relation_type,
                        relation.relation_type,
                    ),
                    "说明": relation.note,
                    "依据": relation.basis,
                }
                for relation in related
            ]
        ),
        hide_index=True,
        width="stretch",
        alt=f"{item.item_key}与其他学习内容的联系",
    )


def _render_item_block(
    service: OutlineService,
    run: dict[str, Any],
    item: Any,
    relations: list[Any],
) -> None:
    with st.container(border=True, key=f"outline_block_{item.id}"):
        heading = st.columns([4, 1], vertical_alignment="center")
        heading[0].markdown(
            f"### {item.ordinal:02d} · {item.content}"
        )
        heading[0].caption(
            f"{item.item_key} · {item.theme_code} {item.theme_name} · "
            f"{ITEM_TYPE_LABELS.get(item.item_type, item.item_type)}"
        )
        heading[1].markdown(
            f":{('green' if item.status == 'verified' else 'gray')}-badge["
            f"{STATUS_LABELS.get(item.status, item.status)}]"
        )
        st.markdown(":blue-badge[内容]")
        st.markdown(item.content)
        if item.key_points:
            for point in item.key_points:
                st.markdown(f"- {point}")
        st.markdown(":green-badge[要求]")
        _render_requirement_sources(item)
        st.markdown(":orange-badge[要求来源]")
        requirements_with_source = [
            requirement
            for requirement in item.requirements
            if requirement.evidence
        ]
        if requirements_with_source:
            for requirement in requirements_with_source:
                evidence = requirement.evidence
                st.caption(
                    f"第 {evidence.page} 页"
                    + (
                        f"（正文第 {evidence.printed_page} 页）"
                        if evidence.printed_page
                        else ""
                    )
                    + f"：{evidence.quote}"
                )
        else:
            st.caption("未附要求来源。")
        st.markdown(":violet-badge[内容来源]")
        _render_item_source_content(item)
        st.markdown(":gray-badge[与其他学习内容的联系]")
        _render_item_relations(item, relations)
        with st.expander("编辑与管理", expanded=False):
            with st.form(f"outline_block_edit_{item.id}"):
                content = st.text_input("学习内容", value=item.content)
                key_points = st.text_area(
                    "学习要点（每行一条）",
                    value="\n".join(item.key_points),
                    height=100,
                )
                indicator_codes = st.text_input(
                    "指标编号（用、分隔）",
                    value="、".join(item.indicator_codes),
                )
                notes = st.text_area("备注", value=item.notes, height=70)
                submitted = st.form_submit_button("保存为新版本")
            if submitted:
                try:
                    service.update_item(
                        item_id=item.id,
                        content=content,
                        key_points=key_points.splitlines(),
                        indicator_codes=[
                            value.strip()
                            for value in indicator_codes.replace("，", "、")
                            .replace(",", "、")
                            .split("、")
                            if value.strip()
                        ],
                        notes=notes,
                        theme_code=item.theme_code,
                        theme_name=item.theme_name,
                        item_type=item.item_type,
                        status=item.status,
                    )
                    _notify("学习内容已保存为新版本。")
                    _rerun()
                except Exception as exc:
                    st.error(str(exc))
            action_columns = st.columns(3)
            if action_columns[0].button(
                "标记已确认",
                key=f"outline_block_verify_{item.id}",
                width="stretch",
            ):
                service.set_items_status(
                    [item.id],
                    OutlineItemStatus.VERIFIED.value,
                )
                _notify("已标记为已确认。")
                _rerun()
            if action_columns[1].button(
                "标记待复核",
                key=f"outline_block_review_{item.id}",
                width="stretch",
            ):
                service.set_items_status(
                    [item.id],
                    OutlineItemStatus.NEEDS_REVIEW.value,
                )
                _notify("已标记为待复核。")
                _rerun()
            if action_columns[2].button(
                "软删除",
                key=f"outline_block_delete_{item.id}",
                width="stretch",
            ):
                service.delete_item(item.id)
                _notify("已软删除该学习内容。")
                _rerun()


def _workflow_rows(workflow: list[Any]) -> list[tuple[str, str]]:
    legacy_labels = {
        "已发送范围与课程标准目录": (
            "范围准备与目录提交",
            "范围与课程标准目录已提交",
        ),
        "第一次 AI 返回需要读取的目录": (
            "目录检索阶段",
            "已返回需要读取的课程标准目录和课本目录",
        ),
        "已获取具体内容": (
            "内容提取阶段",
            "已读取目录检索阶段选中的具体内容",
        ),
        "第二次 AI 返回格式化结果": (
            "结构化生成阶段",
            "已生成学习内容、要求、来源和联系",
        ),
    }
    rows: list[tuple[str, str]] = []
    for index, entry in enumerate(workflow, start=1):
        if isinstance(entry, dict):
            stage = str(entry.get("stage") or f"步骤 {index}")
            detail = str(entry.get("detail") or "")
        else:
            raw = str(entry)
            stage, detail = legacy_labels.get(
                raw,
                (f"步骤 {index}", raw),
            )
        rows.append((stage, detail))
    return rows


def _render_workflow(
    scope: dict[str, Any],
    run_id: str,
) -> None:
    workflow = scope.get("workflow") or []
    if not workflow:
        return
    with st.container(
        border=True,
        key=f"outline_workflow_{run_id}",
    ):
        st.markdown("**处理过程**")
        for index, (stage, detail) in enumerate(
            _workflow_rows(workflow),
            start=1,
        ):
            st.markdown(
                f"{index:02d}. **{stage}**"
                + (f" · {detail}" if detail else "")
            )


def _render_run_blocks(
    service: OutlineService,
    run: dict[str, Any],
) -> None:
    items = service.items(run["id"])
    relations = service.relations(run["id"])
    scope = service.scope_of(run)
    mode_label = SCOPE_MODE_LABELS.get(
        str(scope.get("mode") or ""),
        str(scope.get("mode_label") or "旧格式"),
    )
    st.markdown(
        f"### {mode_label} · 学习内容清单"
    )
    st.caption(scope.get("boundary", {}).get("statement", ""))
    _render_workflow(scope, run["id"])
    if scope.get("specific_requirements"):
        st.caption(
            "具体要求：" + str(scope["specific_requirements"])
        )
    if not items:
        st.info("本次梳理没有学习内容。")
    for item in items:
        _render_item_block(service, run, item, relations)
    with st.expander("导出、字段说明与高级管理", expanded=False):
        _render_export_tab(service, run, items, relations)
        _render_relations_tab(service, run, items, relations)
        if items:
            _render_manual_item(service, run, items[0])


def render_outline_page(
    service: OutlineService,
    profile: ProviderProfile | None,
) -> None:
    service.cleanup_legacy_runs()
    _show_notice()
    _apply_pending_history_run()
    center, settings_column = st.columns([2.7, 1.0], gap="large")
    with settings_column:
        settings, start = _render_outline_settings(service, profile)
    with center:
        runs = service.list_runs()
        selected_run: dict[str, Any] | None = None
        if runs:
            run_ids = [run["id"] for run in runs]
            run_key = "outline_history_run"
            if st.session_state.get(run_key) not in run_ids:
                st.session_state[run_key] = run_ids[0]
            run_id = st.selectbox(
                "历史梳理清单",
                run_ids,
                key=run_key,
                format_func=lambda value: _history_label(
                    service,
                    next(run for run in runs if run["id"] == value),
                ),
            )
            selected_run = next(
                run for run in runs if run["id"] == run_id
            )
        if start and settings and profile:
            result = _run_two_stage_generation(service, settings, profile)
            if result:
                selected_run = result["run"]
        pending = st.session_state.get("_outline_pending_stage")
        if isinstance(pending, dict) and profile and not start:
            if st.button(
                "重试结构化生成阶段",
                type="primary",
                icon=":material/refresh:",
                key="outline_retry_stage_two",
            ):
                process = st.container(
                    border=True,
                    key="outline_retry_process",
                )
                process.markdown("**处理过程**")
                process.markdown(
                    "**01 内容读取** · 正在读取目录检索阶段选中的具体内容"
                )
                try:
                    process.markdown(
                        "**02 结构化生成阶段** · 正在进行内容结构化生成"
                    )
                    result = service.generate_from_selection(
                        scope=pending["scope"],
                        selection=pending["selection"],
                        specific_requirements=pending[
                            "specific_requirements"
                        ],
                        profile=profile,
                    )
                    st.session_state.pop(
                        "_outline_pending_stage",
                        None,
                    )
                    _queue_history_run(result["run"]["id"])
                    selected_run = result["run"]
                    process.markdown(
                        "**03 结果校验与保存** · 已完成来源核对并写入历史清单"
                    )
                    process.success("内容生成阶段重试完成。")
                except Exception as exc:
                    process.markdown("**处理中止**")
                    process.error(str(exc))
        if selected_run is not None:
            _render_run_blocks(service, selected_run)
        elif not start:
            st.info(
                "在右侧选择范围并点击“开始梳理”。"
                "流程将先完成目录检索，再读取所需内容并生成结构化结果。"
            )
