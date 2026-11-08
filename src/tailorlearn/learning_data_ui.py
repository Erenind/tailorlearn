from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import streamlit as st

from tailorlearn.learning_data import (
    DATA_TYPE_SPECS,
    EXTRA_FIELD,
    IGNORE_FIELD,
    DataTypeSpec,
    FieldSpec,
    ImportPreview,
    LearningDataService,
)


ACTION_LABELS = {
    "insert": "新增",
    "update": "更新",
    "skip": "未变化",
    "manual_conflict": "人工修改冲突",
}


def _rerun() -> None:
    st.rerun()


def _show_error(exc: Exception) -> None:
    st.error(str(exc))


def _notify(key: str, message: str) -> None:
    st.session_state[key] = message


def _show_notice(key: str) -> None:
    message = st.session_state.pop(key, None)
    if message:
        st.success(message)


def _field_input(
    field_spec: FieldSpec,
    *,
    value: Any,
    key: str,
) -> Any:
    text_value = "" if value is None else str(value)
    if field_spec.multiline:
        return st.text_area(
            field_spec.label,
            value=text_value,
            height=120,
            key=key,
            help=(
                "支持多行文本"
                if field_spec.value_type == "text"
                else field_spec.value_type
            ),
        )
    return st.text_input(
        field_spec.label,
        value=text_value,
        key=key,
        help=(
            "YYYY-MM-DD"
            if field_spec.value_type == "date"
            else (
                "ISO 8601，例如 2026-01-20T08:56:22+08:00"
                if field_spec.value_type == "datetime"
                else (
                    "整数"
                    if field_spec.value_type == "integer"
                    else None
                )
            )
        ),
    )


def _record_form_values(spec: DataTypeSpec) -> dict[str, Any]:
    return {
        item.key: _field_input(
            item,
            value=None,
            key=f"learning_new_{spec.key}_{item.key}",
        )
        for item in spec.fields
    }


def _render_expected_fields(spec: DataTypeSpec, service: LearningDataService) -> None:
    st.dataframe(
        pd.DataFrame(service.field_definition_rows(spec.key)),
        hide_index=True,
        width="stretch",
        alt=f"{spec.label}导入字段说明",
    )


def _render_alias_manager(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
) -> None:
    aliases = service.aliases(dataset_id, spec.key)
    if not aliases:
        st.caption("尚未保存字段别名。")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "源字段": row["source_name"],
                    "映射到": service.mapping_label(
                        spec.key,
                        row["target_field"],
                    ),
                }
                for row in aliases
            ]
        ),
        hide_index=True,
        width="stretch",
        alt=f"{spec.label}已保存字段别名",
    )
    source_name = st.selectbox(
        "要删除的字段别名",
        [row["source_name"] for row in aliases],
        key=f"learning_alias_delete_select_{dataset_id}_{spec.key}",
    )
    if st.button(
        "删除该映射规则",
        key=f"learning_alias_delete_{dataset_id}_{spec.key}",
    ):
        service.delete_alias(dataset_id, spec.key, source_name)
        _notify(
            f"learning_notice_{dataset_id}_{spec.key}",
            "字段映射规则已删除",
        )
        _rerun()


def _render_mapping(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
    file: Any,
) -> dict[str, str]:
    content = file.getvalue()
    file_signature = hashlib.sha256(content).hexdigest()[:12]
    table = service.read_csv(content)
    suggested = service.suggest_mapping(
        dataset_id,
        spec.key,
        table.headers,
    )
    targets = [*service.valid_targets(spec.key), EXTRA_FIELD, IGNORE_FIELD]
    mapping: dict[str, str] = {}
    st.markdown("**字段映射**")
    st.caption(
        "文件名、列顺序和 CSV 编码不影响识别。自动匹配不上的列默认作为扩展字段保留。"
    )
    for index, header in enumerate(table.headers):
        target = suggested[header]
        default_index = targets.index(target) if target in targets else len(targets) - 2
        selected = st.selectbox(
            f"源字段：{header}",
            targets,
            index=default_index,
            format_func=lambda value: service.mapping_label(
                spec.key,
                value,
            ),
            key=(
                f"learning_mapping_{dataset_id}_{spec.key}_"
                f"{file_signature}_{index}"
            ),
        )
        mapping[header] = selected
    return mapping


def _preview_frame(preview: ImportPreview) -> pd.DataFrame:
    spec = preview.spec
    rows = [
        {
            "动作": ACTION_LABELS.get(row["_action"], row["_action"]),
            **{
                item.label: (
                    ""
                    if row.get(item.key) is None
                    else str(row.get(item.key))
                )
                for item in spec.fields
            },
        }
        for row in preview.rows
    ]
    return pd.DataFrame(rows)


def _render_import(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
    *,
    has_records: bool,
) -> None:
    notice_key = f"learning_notice_{dataset_id}_{spec.key}"
    _show_notice(notice_key)
    with st.expander("导入说明与标准字段", expanded=not has_records):
        st.caption(
            "每次只导入一个同类型 CSV。先预览新增、更新、未变化、"
            "人工修改冲突和业务质量问题，再确认写入。"
        )
        _render_expected_fields(spec, service)

    file = st.file_uploader(
        f"上传{spec.label} CSV",
        type=["csv"],
        key=f"learning_upload_{dataset_id}_{spec.key}",
        help="仅选择一个 CSV 文件；重复导入请再次操作。",
    )
    with st.expander("已保存字段映射"):
        _render_alias_manager(service, dataset_id, spec)
    if file is None:
        return
    try:
        mapping = _render_mapping(service, dataset_id, spec, file)
        errors = service.mapping_errors(spec.key, mapping)
        overwrite_manual = st.checkbox(
            "覆盖已人工修改的冲突记录",
            value=False,
            key=f"learning_overwrite_{dataset_id}_{spec.key}",
            help="未勾选时，人工修改过的记录默认保留当前值。",
        )
        preview = service.preview_import(
            dataset_id,
            spec.key,
            file.name,
            file.getvalue(),
            mapping,
            overwrite_manual=overwrite_manual,
        )
    except Exception as exc:
        _show_error(exc)
        return

    errors = list(dict.fromkeys([*errors, *preview.errors]))
    if errors:
        st.error("导入前需要修正以下结构问题：")
        for error in errors[:20]:
            st.write(f"- {error}")
        if len(errors) > 20:
            st.caption(f"另有 {len(errors) - 20} 个同类问题未展开。")
        return

    metrics = st.columns(5)
    metrics[0].metric("总行数", preview.counts.get("total", 0))
    metrics[1].metric("新增", preview.counts.get("insert", 0))
    metrics[2].metric("更新/恢复", preview.counts.get("update", 0))
    metrics[3].metric("未变化", preview.counts.get("skip", 0))
    metrics[4].metric(
        "人工修改冲突",
        preview.counts.get("manual_conflict", 0),
    )
    if preview.counts.get("manual_conflict"):
        st.warning(
            "检测到人工修改冲突；未勾选覆盖时这些行会跳过，"
            "其余行仍按预览结果导入。"
        )
    if preview.rows:
        with st.expander("逐行预览", expanded=True):
            st.dataframe(
                _preview_frame(preview),
                hide_index=True,
                width="stretch",
                alt=f"{spec.label}导入逐行预览",
            )
    if preview.issues:
        st.warning(f"发现 {len(preview.issues)} 个业务质量问题，可随数据导入并标记。")
        st.dataframe(
            pd.DataFrame(preview.issues).head(100),
            hide_index=True,
            width="stretch",
            alt=f"{spec.label}导入质量提示",
        )
    remember_mapping = st.checkbox(
        "记住本次字段映射",
        value=False,
        key=f"learning_remember_mapping_{dataset_id}_{spec.key}",
    )
    if st.button(
        f"确认导入{spec.label}",
        type="primary",
        key=f"learning_confirm_import_{dataset_id}_{spec.key}",
        width="stretch",
    ):
        try:
            result = service.execute_import(
                preview,
                remember_mapping=remember_mapping,
            )
            _notify(
                notice_key,
                f"导入完成：新增 {result.inserted}，更新 {result.updated}，"
                f"跳过 {result.skipped}，质量问题 {result.issues}。",
            )
            _rerun()
        except Exception as exc:
            _show_error(exc)


def _matches_query(
    row: dict[str, Any],
    spec: DataTypeSpec,
    query: str,
) -> bool:
    if not query:
        return True
    needle = query.casefold().strip()
    return any(
        needle in str(row.get(item.key, "")).casefold()
        for item in spec.fields
    )


def _visible_records(
    rows: list[dict[str, Any]],
    spec: DataTypeSpec,
    query: str,
    answer_status: str | None,
    answer_node: str | None,
) -> list[dict[str, Any]]:
    visible = [row for row in rows if _matches_query(row, spec, query)]
    if spec.key == "answer" and answer_status and answer_status != "全部":
        visible = [
            row for row in visible if row.get("grading_status") == answer_status
        ]
    if spec.key == "answer" and answer_node and answer_node != "全部":
        visible = [
            row for row in visible if row.get("node_code") == answer_node
        ]
    return visible


def _records_frame(
    rows: list[dict[str, Any]],
    spec: DataTypeSpec,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **{
                    item.label: (
                        ""
                        if row.get(item.key) is None
                        else str(row.get(item.key))
                    )
                    for item in spec.fields
                },
                "状态": "已删除" if row.get("deleted_at") else "正常",
            }
            for row in rows
        ]
    )


def _render_add_form(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
) -> None:
    with st.expander("新增记录"):
        with st.form(
            f"learning_add_{dataset_id}_{spec.key}",
            border=False,
        ):
            values = _record_form_values(spec)
            submitted = st.form_submit_button(
                "新增",
                type="primary",
                icon=":material/add:",
            )
        if submitted:
            try:
                service.create_record(dataset_id, spec.key, values)
                _notify(
                    f"learning_notice_{dataset_id}_{spec.key}",
                    f"已新增一条{spec.label}记录",
                )
                _rerun()
            except Exception as exc:
                _show_error(exc)


def _render_edit_forms(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
    row: dict[str, Any],
) -> None:
    row_id = int(row["row_id"])
    with st.expander("编辑普通字段", expanded=False):
        with st.form(
            f"learning_edit_{dataset_id}_{spec.key}_{row_id}",
            border=False,
        ):
            values: dict[str, Any] = {}
            for item in spec.fields:
                if item.key in spec.natural_key:
                    continue
                values[item.key] = _field_input(
                    item,
                    value=row.get(item.key),
                    key=(
                        f"learning_edit_input_{dataset_id}_{spec.key}_"
                        f"{row_id}_{item.key}"
                    ),
                )
            submitted = st.form_submit_button(
                "保存修改",
                type="primary",
                icon=":material/save:",
            )
        if submitted:
            try:
                service.update_record(
                    dataset_id,
                    spec.key,
                    row_id,
                    values,
                )
                _notify(
                    f"learning_notice_{dataset_id}_{spec.key}",
                    "记录已更新，并保留修改审计",
                )
                _rerun()
            except Exception as exc:
                _show_error(exc)

    with st.expander("修改业务编号"):
        st.caption(
            "业务编号变更会影响后续文件匹配；关联作答按内部记录身份保持连接。"
        )
        with st.form(
            f"learning_rekey_{dataset_id}_{spec.key}_{row_id}",
            border=False,
        ):
            identity_values = {
                item.key: _field_input(
                    item,
                    value=row.get(item.key),
                    key=(
                        f"learning_rekey_input_{dataset_id}_{spec.key}_"
                        f"{row_id}_{item.key}"
                    ),
                )
                for item in spec.fields
                if item.key in spec.natural_key
            }
            confirmed = st.checkbox(
                "我确认修改业务编号",
                key=(
                    f"learning_rekey_confirm_{dataset_id}_{spec.key}_{row_id}"
                ),
            )
            submitted = st.form_submit_button(
                "保存业务编号",
                icon=":material/key:",
            )
        if submitted:
            if not confirmed:
                st.error("请先确认业务编号修改")
            else:
                try:
                    service.update_record(
                        dataset_id,
                        spec.key,
                        row_id,
                        identity_values,
                        allow_identity_change=True,
                    )
                    _notify(
                        f"learning_notice_{dataset_id}_{spec.key}",
                        "业务编号已更新",
                    )
                    _rerun()
                except Exception as exc:
                    _show_error(exc)


def _render_delete_restore(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
    row: dict[str, Any],
) -> None:
    row_id = int(row["row_id"])
    related = service.related_counts(dataset_id, spec.key, row)
    if row.get("deleted_at"):
        with st.expander("恢复记录"):
            if related:
                st.caption(
                    "关联记录："
                    + "，".join(
                        f"{label} {count} 条" for label, count in related.items()
                    )
                )
            cascade = st.checkbox(
                "同时恢复关联作答记录",
                value=bool(related),
                key=f"learning_restore_cascade_{dataset_id}_{spec.key}_{row_id}",
                disabled=not related,
            )
            if st.button(
                "恢复所选记录",
                key=f"learning_restore_{dataset_id}_{spec.key}_{row_id}",
            ):
                try:
                    count = service.restore_record(
                        dataset_id,
                        spec.key,
                        row_id,
                        cascade=cascade,
                    )
                    _notify(
                        f"learning_notice_{dataset_id}_{spec.key}",
                        f"已恢复 {count} 条记录",
                    )
                    _rerun()
                except Exception as exc:
                    _show_error(exc)
        return

    with st.expander("删除记录"):
        if related:
            st.warning(
                "关联记录："
                + "，".join(
                    f"{label} {count} 条" for label, count in related.items()
                )
            )
        cascade = st.checkbox(
            "同时软删除关联作答记录",
            value=bool(related),
            key=f"learning_delete_cascade_{dataset_id}_{spec.key}_{row_id}",
            disabled=not related,
        )
        confirmed = st.checkbox(
            "我确认删除所选记录",
            key=f"learning_delete_confirm_{dataset_id}_{spec.key}_{row_id}",
        )
        if st.button(
            "软删除",
            key=f"learning_delete_{dataset_id}_{spec.key}_{row_id}",
        ):
            if not confirmed:
                st.error("请先确认删除")
            else:
                try:
                    count = service.soft_delete_record(
                        dataset_id,
                        spec.key,
                        row_id,
                        cascade=cascade,
                    )
                    _notify(
                        f"learning_notice_{dataset_id}_{spec.key}",
                        f"已软删除 {count} 条记录，可在已删除视图中恢复",
                    )
                    _rerun()
                except Exception as exc:
                    _show_error(exc)


def _render_records(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
) -> None:
    all_active = service.list_records(dataset_id, spec.key)
    st.subheader("数据管理与导出")
    export_columns = st.columns(2)
    full_csv, filename = service.export_csv(dataset_id, spec.key)
    export_columns[0].download_button(
        "导出全部",
        data=full_csv,
        file_name=filename,
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
        disabled=not all_active,
    )
    st.caption(f"共 {len(all_active)} 条有效记录。")

    search_columns = st.columns([2, 1, 1, 1])
    query = search_columns[0].text_input(
        "搜索",
        type="search",
        placeholder="输入编号或任意字段内容",
        label_visibility="collapsed",
        key=f"learning_search_{dataset_id}_{spec.key}",
    )
    latest_answers = False
    status = None
    node_code = None
    if spec.key == "answer":
        latest_answers = search_columns[1].toggle(
            "仅最新修订",
            value=False,
            key=f"learning_latest_{dataset_id}_{spec.key}",
        )
        status = search_columns[2].selectbox(
            "评分状态",
            ["全部", "已评分", "待评分", "未作答"],
            key=f"learning_status_{dataset_id}_{spec.key}",
        )
        nodes = sorted(
            {
                row.get("node_code")
                for row in all_active
                if row.get("node_code")
            }
        )
        node_code = search_columns[3].selectbox(
            "节点",
            ["全部", *nodes],
            key=f"learning_node_filter_{dataset_id}_{spec.key}",
        )

    active_rows = service.list_records(
        dataset_id,
        spec.key,
        latest_answers=latest_answers,
    )
    visible = _visible_records(
        active_rows,
        spec,
        query,
        status,
        node_code,
    )
    filtered_csv, filtered_filename = service.export_csv(
        dataset_id,
        spec.key,
        rows=visible,
    )
    export_columns[1].download_button(
        "导出当前结果",
        data=filtered_csv,
        file_name=filtered_filename,
        mime="text/csv",
        icon=":material/filter_alt:",
        width="stretch",
        disabled=not visible,
    )
    include_deleted = st.toggle(
        "显示已删除记录",
        value=False,
        key=f"learning_show_deleted_{dataset_id}_{spec.key}",
    )
    display_rows = visible
    if include_deleted:
        deleted_rows = [
            row
            for row in service.list_records(
                dataset_id,
                spec.key,
                include_deleted=True,
            )
            if row.get("deleted_at")
            and _matches_query(row, spec, query)
        ]
        display_rows = [*visible, *deleted_rows]
    st.caption(f"当前筛选显示 {len(display_rows)} 条记录。")
    if display_rows:
        st.dataframe(
            _records_frame(display_rows, spec),
            hide_index=True,
            width="stretch",
            alt=f"{spec.label}数据记录",
        )
    else:
        st.info("当前筛选条件下没有记录。")

    _render_add_form(service, dataset_id, spec)
    if not display_rows:
        return
    row_ids = [int(row["row_id"]) for row in display_rows]
    row_by_id = {int(row["row_id"]): row for row in display_rows}
    selector_key = f"learning_record_select_{dataset_id}_{spec.key}"
    if st.session_state.get(selector_key) not in row_ids:
        st.session_state[selector_key] = row_ids[0]
    selected_id = st.selectbox(
        "选择要操作的记录",
        row_ids,
        format_func=lambda value: service.record_label(
            spec.key,
            row_by_id[value],
        ),
        key=selector_key,
    )
    selected = row_by_id[selected_id]
    _render_delete_restore(service, dataset_id, spec, selected)
    if not selected.get("deleted_at"):
        _render_edit_forms(service, dataset_id, spec, selected)


def _render_type(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
) -> None:
    active_count = len(service.list_records(dataset_id, spec.key))
    st.subheader(spec.label)
    if spec.key == "answer":
        st.caption(
            "原始修订全部保留。表格与导出默认显示全部记录，"
            "可切换“仅最新修订”。"
        )
    _render_import(
        service,
        dataset_id,
        spec,
        has_records=bool(active_count),
    )
    st.divider()
    _render_records(service, dataset_id, spec)


def _render_quality(
    service: LearningDataService,
    dataset_id: str,
) -> None:
    st.subheader("数据质量")
    st.caption(
        "只列出确定性检查结果。信息不足会明确标为待补充，"
        "不会直接解释为未掌握。"
    )
    report = service.quality_report(dataset_id)
    counts = report["counts"]
    metric_columns = st.columns(6)
    for column, (label, count) in zip(
        metric_columns,
        counts.items(),
        strict=True,
    ):
        column.metric(label, count)
    if report["issues"]:
        st.warning(f"发现 {len(report['issues'])} 个当前数据质量问题。")
        st.dataframe(
            pd.DataFrame(report["issues"]),
            hide_index=True,
            width="stretch",
            alt="学情数据质量问题清单",
        )
    else:
        st.success("当前数据未发现已定义的质量问题。")

    distributions = report["distributions"]
    distribution_columns = st.columns(2)
    distribution_columns[0].markdown("**学生班级分布**")
    distribution_columns[0].dataframe(
        pd.DataFrame(
            [
                {"班级": key, "人数": value}
                for key, value in distributions["class"].items()
            ]
        ),
        hide_index=True,
        width="stretch",
        alt="学生班级分布",
    )
    distribution_columns[1].markdown("**作答评分状态分布**")
    distribution_columns[1].dataframe(
        pd.DataFrame(
            [
                {"评分状态": key, "记录数": value}
                for key, value in distributions["grading_status"].items()
            ]
        ),
        hide_index=True,
        width="stretch",
        alt="作答评分状态分布",
    )
    batches = service.list_import_batches(dataset_id)
    with st.expander(f"导入批次（{len(batches)}）"):
        if not batches:
            st.caption("尚无导入批次。")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "批次": row["id"],
                            "类型": DATA_TYPE_SPECS[row["data_type"]].label,
                            "文件": row["file_name"],
                            "状态": row["status"],
                            "总行数": row["total_rows"],
                            "新增": row["inserted_rows"],
                            "更新": row["updated_rows"],
                            "跳过": row["skipped_rows"],
                            "冲突": row["conflict_rows"],
                            "问题": row["issue_count"],
                            "时间": row["created_at"],
                        }
                        for row in batches
                    ]
                ),
                hide_index=True,
                width="stretch",
                alt="学情数据导入批次记录",
            )


def _render_dataset_management(
    service: LearningDataService,
    dataset: dict[str, Any],
) -> None:
    with st.expander("编辑数据集信息"):
        with st.form(f"learning_dataset_edit_{dataset['id']}"):
            name = st.text_input("数据集名称", value=dataset["name"])
            description = st.text_area(
                "说明",
                value=dataset.get("description") or "",
                height=80,
            )
            submitted = st.form_submit_button("保存数据集信息")
        if submitted:
            try:
                service.update_dataset(
                    dataset["id"],
                    name=name,
                    description=description,
                )
                _notify("learning_dataset_notice", "数据集信息已保存")
                _rerun()
            except Exception as exc:
                _show_error(exc)


def render_learning_data_page(service: LearningDataService) -> None:
    st.title("学情数据")
    st.caption(
        "按数据集隔离六类赛题 CSV，支持独立导入导出、"
        "重复合并、可恢复删除和确定性质量检查。"
    )
    _show_notice("learning_dataset_notice")
    with st.expander("新建学情数据集", expanded=not service.list_datasets()):
        with st.form("learning_dataset_create"):
            name = st.text_input(
                "数据集名称",
                placeholder="例如：公开模拟学情数据",
            )
            description = st.text_area(
                "说明",
                placeholder="可记录数据来源、批次或使用范围",
                height=80,
            )
            submitted = st.form_submit_button(
                "创建数据集",
                type="primary",
                icon=":material/create_new_folder:",
            )
        if submitted:
            try:
                dataset = service.create_dataset(name, description)
                st.session_state["learning_dataset_selector"] = dataset["id"]
                _notify("learning_dataset_notice", "学情数据集已创建")
                _rerun()
            except Exception as exc:
                _show_error(exc)

    datasets = service.list_datasets()
    if not datasets:
        st.info("请先创建一个学情数据集。")
        return
    dataset_ids = [row["id"] for row in datasets]
    selector_key = "learning_dataset_selector"
    if st.session_state.get(selector_key) not in dataset_ids:
        st.session_state[selector_key] = dataset_ids[0]
    dataset_id = st.selectbox(
        "学情数据集",
        dataset_ids,
        format_func=lambda value: next(
            (
                row["name"]
                + (
                    f" · {row['description']}"
                    if row.get("description")
                    else ""
                )
                for row in datasets
                if row["id"] == value
            ),
            value,
        ),
        key=selector_key,
    )
    dataset = next(row for row in datasets if row["id"] == dataset_id)
    _render_dataset_management(service, dataset)

    tabs = st.tabs(
        [
            *[spec.label for spec in DATA_TYPE_SPECS.values()],
            "数据质量",
        ]
    )
    for tab, spec in zip(
        tabs[:-1],
        DATA_TYPE_SPECS.values(),
        strict=True,
    ):
        with tab:
            _render_type(service, dataset_id, spec)
    with tabs[-1]:
        _render_quality(service, dataset_id)
