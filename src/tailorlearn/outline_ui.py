from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from tailorlearn.models import OutlineItemStatus, ProviderProfile
from tailorlearn.outline import (
    ITEM_TYPE_LABELS,
    OUTLINE_FIELD_DEFINITIONS,
    RELATION_TYPE_LABELS,
    SOURCE_KIND_LABELS,
    STATUS_LABELS,
    OutlineService,
)


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
    return (
        f"{prefix}{row['file_name']} · {row['chapter']} / {row['title']}"
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


def render_outline_page(
    service: OutlineService,
    profile: ProviderProfile | None,
) -> None:
    _show_notice()
    datasets = service.list_datasets()
    if not datasets:
        st.info("请先在“材料中心 → 课程要求”中创建数据集并导入课程要求。")
        return
    dataset_ids = [row["id"] for row in datasets]
    dataset_key = "outline_dataset_selector"
    latest_runs = service.list_runs()
    preferred_dataset = (
        latest_runs[0]["dataset_id"] if latest_runs else dataset_ids[0]
    )
    if (
        st.session_state.get(dataset_key) not in dataset_ids
        or dataset_key not in st.session_state
    ):
        st.session_state[dataset_key] = (
            preferred_dataset
            if preferred_dataset in dataset_ids
            else dataset_ids[0]
        )
    dataset_id = st.selectbox(
        "学情数据集（提供主题、指标与范围依据）",
        dataset_ids,
        key=dataset_key,
        format_func=lambda value: next(
            (
                f"{row['name']}"
                + (f" · {row['description']}" if row.get("description") else "")
                for row in datasets
                if row["id"] == value
            ),
            value,
        ),
    )
    requirements = service.list_requirements(dataset_id)
    if not requirements:
        st.warning(
            "该数据集没有课程要求，无法确定范围。"
            "请先在“材料中心 → 课程要求”导入。"
        )
        return
    (
        themes,
        selected_themes,
        standard_material,
        standard_document,
        textbook_materials,
        textbook_documents,
    ) = _render_selectors(service, dataset_id)
    textbook_material = textbook_materials[0] if textbook_materials else None
    scope = _render_scope_editor(
        service,
        dataset_id,
        selected_themes,
        standard_material,
        standard_document,
        textbook_material,
        textbook_documents,
        themes,
    )
    if scope is None:
        return
    _render_generate(service, scope, profile)
    st.divider()
    runs = service.list_runs(dataset_id)
    if not runs:
        st.info("尚未生成梳理结果。确认范围后点击“开始规则梳理”。")
        return
    run_ids = [row["id"] for row in runs]
    run_key = f"outline_run_selector_{dataset_id}"
    if st.session_state.get(run_key) not in run_ids:
        st.session_state[run_key] = run_ids[0]
    run_id = st.selectbox(
        "梳理记录",
        run_ids,
        key=run_key,
        format_func=lambda value: next(
            (
                f"{row['name']} · {row['item_count']} 条 · {row['created_at']}"
                for row in runs
                if row["id"] == value
            ),
            value,
        ),
    )
    run = next(row for row in runs if row["id"] == run_id)
    items = service.items(run_id)
    relations = service.relations(run_id)
    _run_metrics(run, items, relations)
    tabs = st.tabs(
        ["学习内容与要求", "必要联系", "材料出处与核对", "导出与字段说明"]
    )
    with tabs[0]:
        _render_items_tab(service, run, items, relations, profile)
    with tabs[1]:
        _render_relations_tab(service, run, items, relations)
    with tabs[2]:
        _render_sources_tab(service, run, items, relations)
    with tabs[3]:
        _render_export_tab(service, run, items, relations)
