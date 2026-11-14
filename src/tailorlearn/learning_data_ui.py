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

LEARNING_DATA_UI_VERSION = "2026-11-14-import-direct-create-v2"


ACTION_LABELS = {
    "insert": "新增",
    "update": "更新",
    "skip": "未变化",
    "manual_conflict": "人工修改冲突",
}


def _apply_table_action(
    *,
    click_key: str,
    row_ids: list[int],
    action_key: str,
    action: str,
) -> None:
    click = st.session_state.get(click_key)
    if click is None:
        return
    try:
        row_position = int(click["row"])
    except (KeyError, TypeError, ValueError):
        return
    if not 0 <= row_position < len(row_ids):
        return
    st.session_state[action_key] = {
        "action": action,
        "row_id": int(row_ids[row_position]),
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


def _render_data_export(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
) -> None:
    rows = service.list_records(
        dataset_id,
        spec.key,
        latest_answers=spec.key == "answer",
    )
    with st.expander(
        f"导出{spec.label}",
        expanded=bool(rows),
        icon=":material/download:",
    ):
        if not rows:
            st.caption(f"暂无可导出的{spec.label}数据。")
            return
        csv_content, filename = service.export_csv(
            dataset_id,
            spec.key,
            rows=rows,
        )
        suffix = "（仅最新修订）" if spec.key == "answer" else ""
        st.caption(f"将导出 {len(rows)} 条记录{suffix}。")
        st.download_button(
            f"下载{spec.label} CSV",
            data=csv_content,
            file_name=filename,
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
            key=f"learning_export_{dataset_id}_{spec.key}",
        )


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
    *,
    with_actions: bool = False,
) -> pd.DataFrame:
    records = []
    for row in rows:
        record = {
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
        if with_actions:
            record["编辑"] = ":material/edit: 编辑"
            record["删除"] = ":material/delete: 删除"
        records.append(record)
    return pd.DataFrame(records)


@st.dialog("新增记录", width="large")
def _add_record_dialog(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
) -> None:
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
            st.session_state.pop(
                f"learning_add_open_{dataset_id}_{spec.key}",
                None,
            )
            _notify(
                f"learning_notice_{dataset_id}_{spec.key}",
                f"已新增一条{spec.label}记录",
            )
            _rerun()
        except Exception as exc:
            _show_error(exc)
    if st.button(
        "取消",
        key=f"learning_add_cancel_{dataset_id}_{spec.key}",
        width="stretch",
    ):
        st.session_state.pop(
            f"learning_add_open_{dataset_id}_{spec.key}",
            None,
        )
        st.rerun()


def _render_add_form(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
) -> None:
    add_key = f"learning_add_open_{dataset_id}_{spec.key}"
    if st.button(
        "新增记录",
        icon=":material/add:",
        key=f"learning_add_toggle_{dataset_id}_{spec.key}",
    ):
        st.session_state[add_key] = True
        st.rerun()
    if st.session_state.get(add_key):
        _add_record_dialog(service, dataset_id, spec)


def _render_edit_forms(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
    row: dict[str, Any],
    *,
    action_key: str | None = None,
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
                if action_key:
                    st.session_state.pop(action_key, None)
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
                    if action_key:
                        st.session_state.pop(action_key, None)
                    _rerun()
                except Exception as exc:
                    _show_error(exc)


def _render_delete_restore(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
    row: dict[str, Any],
    *,
    action_key: str | None = None,
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
                    if action_key:
                        st.session_state.pop(action_key, None)
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
                    if action_key:
                        st.session_state.pop(action_key, None)
                    _rerun()
                except Exception as exc:
                    _show_error(exc)


@st.dialog("记录操作", width="large")
def _record_action_dialog(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
    *,
    action_key: str,
) -> None:
    action = st.session_state.get(action_key)
    if not isinstance(action, dict):
        return
    row_id = action.get("row_id")
    if row_id is None:
        st.session_state.pop(action_key, None)
        return
    row = service.get_record(dataset_id, spec.key, int(row_id))
    if row is None:
        st.session_state.pop(action_key, None)
        st.info("该记录已不存在。")
        return
    st.caption(f"正在处理：{service.record_label(spec.key, row)}")
    if st.button(
        "关闭",
        key=f"{action_key}_close",
        width="stretch",
    ):
        st.session_state.pop(action_key, None)
        st.rerun()
    if action.get("action") == "edit" and not row.get("deleted_at"):
        _render_edit_forms(
            service,
            dataset_id,
            spec,
            row,
            action_key=action_key,
        )
    else:
        _render_delete_restore(
            service,
            dataset_id,
            spec,
            row,
            action_key=action_key,
        )


def _render_record_action_panel(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
    *,
    action_key: str,
) -> None:
    if isinstance(st.session_state.get(action_key), dict):
        _record_action_dialog(service, dataset_id, spec, action_key=action_key)


def _render_records(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
    *,
    allow_edit: bool,
) -> None:
    all_active = service.list_records(dataset_id, spec.key)
    query = ""
    include_deleted = False
    latest_answers = False
    status = None
    node_code = None
    if allow_edit:
        toolbar = st.columns([5, 1, 1], vertical_alignment="bottom")
        query = toolbar[0].text_input(
            "搜索",
            type="search",
            placeholder="输入编号或任意字段内容",
            label_visibility="collapsed",
            key=f"learning_search_{dataset_id}_{spec.key}",
        )
        include_deleted = toolbar[1].toggle(
            "显示已删除",
            value=False,
            key=f"learning_show_deleted_{dataset_id}_{spec.key}",
        )
        with toolbar[2]:
            _render_add_form(service, dataset_id, spec)

    if spec.key == "answer" and allow_edit:
        answer_filters = st.columns(3)
        latest_answers = answer_filters[0].toggle(
            "仅最新修订",
            value=False,
            key=f"learning_latest_{dataset_id}_{spec.key}",
        )
        status = answer_filters[1].selectbox(
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
        node_code = answer_filters[2].selectbox(
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
    action_key = f"learning_record_action_{dataset_id}_{spec.key}"
    if display_rows:
        table = _records_frame(
            display_rows,
            spec,
            with_actions=allow_edit,
        )
        if allow_edit:
            row_ids = [int(row["row_id"]) for row in display_rows]
            edit_click_key = (
                f"learning_record_edit_click_{dataset_id}_{spec.key}"
            )
            delete_click_key = (
                f"learning_record_delete_click_{dataset_id}_{spec.key}"
            )
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                height="content",
                alt=f"{spec.label}数据记录",
                column_config={
                    "编辑": st.column_config.ButtonColumn(
                        type="tertiary",
                        on_click=_apply_table_action,
                        kwargs={
                            "click_key": edit_click_key,
                            "row_ids": row_ids,
                            "action_key": action_key,
                            "action": "edit",
                        },
                        key=edit_click_key,
                    ),
                    "删除": st.column_config.ButtonColumn(
                        type="tertiary",
                        on_click=_apply_table_action,
                        kwargs={
                            "click_key": delete_click_key,
                            "row_ids": row_ids,
                            "action_key": action_key,
                            "action": "delete",
                        },
                        key=delete_click_key,
                    ),
                },
            )
        else:
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                height="content",
                alt=f"{spec.label}数据记录",
            )
    else:
        st.info("当前筛选条件下没有记录。")

    if not allow_edit:
        return
    if not display_rows:
        return
    _render_record_action_panel(
        service,
        dataset_id,
        spec,
        action_key=action_key,
    )


def _render_type(
    service: LearningDataService,
    dataset_id: str,
    spec: DataTypeSpec,
    *,
    mode: str,
) -> None:
    if mode not in {"browse", "edit", "import"}:
        raise ValueError("未知材料与数据栏目")
    if mode == "browse":
        st.caption("浏览模式：仅查看当前数据集的已有记录，不提供新增、编辑或删除。")
    elif mode == "edit":
        st.caption("编辑模式：可新增、编辑、软删除和恢复当前数据集记录。")
    else:
        st.caption("导入和导出：新建数据集、导入 CSV 或下载导出文件。")
    if spec.key == "answer":
        if mode != "browse":
            st.caption(
                "原始修订全部保留。表格与导出默认显示全部记录，"
                "可切换“仅最新修订”。"
            )
    if mode == "import":
        _render_data_export(service, dataset_id, spec)
        active_count = len(service.list_records(dataset_id, spec.key))
        _render_import(
            service,
            dataset_id,
            spec,
            has_records=bool(active_count),
        )
        return
    _render_records(
        service,
        dataset_id,
        spec,
        allow_edit=mode == "edit",
    )


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


def render_answer_anomalies(
    service: LearningDataService,
    dataset_id: str,
) -> None:
    st.subheader("作答记录异常检测")
    st.caption(
        "检查评分状态、得分范围、关联引用、修订版本和评分一致性；"
        "异常记录只提示需要核对，不直接解释为未掌握。"
    )
    report = service.quality_report(dataset_id)
    answer_issues = [
        row for row in report["issues"] if row.get("数据类型") == "作答记录"
    ]
    answers = service.list_records(dataset_id, "answer")
    pending = sum(row.get("grading_status") == "待评分" for row in answers)
    unanswered = sum(row.get("grading_status") == "未作答" for row in answers)
    affected_records = {
        str(row.get("记录")) for row in answer_issues if row.get("记录")
    }
    summary = st.columns(4)
    summary[0].metric("作答记录", len(answers), border=True)
    summary[1].metric("异常项", len(answer_issues), border=True)
    summary[2].metric("涉及记录", len(affected_records), border=True)
    summary[3].metric("待评分/未作答", pending + unanswered, border=True)
    if not answer_issues:
        st.success("当前作答记录未发现已定义的异常。")
        return
    query = st.text_input(
        "搜索异常记录",
        type="search",
        placeholder="输入记录、学生、材料或问题说明",
        key=f"learning_answer_anomaly_search_{dataset_id}",
    )
    needle = query.casefold().strip()
    visible = [
        row
        for row in answer_issues
        if not needle
        or any(
            needle in str(value).casefold()
            for value in row.values()
        )
    ]
    issue_types = pd.DataFrame(answer_issues)["问题代码"].value_counts()
    st.dataframe(
        issue_types.rename_axis("问题代码").reset_index(name="数量"),
        hide_index=True,
        width="stretch",
        alt="作答记录异常类型分布",
    )
    st.caption(f"当前显示 {len(visible)} / {len(answer_issues)} 个异常项。")
    if visible:
        row_ids = [
            int(row["row_id"]) if row.get("row_id") is not None else -1
            for row in visible
        ]
        action_key = f"learning_answer_anomaly_action_{dataset_id}"
        edit_click_key = f"learning_answer_anomaly_edit_click_{dataset_id}"
        delete_click_key = f"learning_answer_anomaly_delete_click_{dataset_id}"
        issue_frame = pd.DataFrame(visible)
        issue_frame["编辑"] = ":material/edit: 编辑"
        issue_frame["删除"] = ":material/delete: 删除"
        st.dataframe(
            issue_frame,
            hide_index=True,
            width="stretch",
            height="content",
            alt="作答记录异常明细",
            column_config={
                "row_id": None,
                "编辑": st.column_config.ButtonColumn(
                    type="tertiary",
                    on_click=_apply_table_action,
                    kwargs={
                        "click_key": edit_click_key,
                        "row_ids": row_ids,
                        "action_key": action_key,
                        "action": "edit",
                    },
                    key=edit_click_key,
                ),
                "删除": st.column_config.ButtonColumn(
                    type="tertiary",
                    on_click=_apply_table_action,
                    kwargs={
                        "click_key": delete_click_key,
                        "row_ids": row_ids,
                        "action_key": action_key,
                        "action": "delete",
                    },
                    key=delete_click_key,
                ),
            },
        )
        st.caption("使用异常明细末列的“编辑”或“删除”处理对应作答记录。")
        _render_record_action_panel(
            service,
            dataset_id,
            DATA_TYPE_SPECS["answer"],
            action_key=action_key,
        )
    else:
        st.info("没有匹配当前搜索条件的异常记录。")


@st.dialog("编辑数据集信息", width="medium")
def _dataset_info_dialog(
    service: LearningDataService,
    dataset: dict[str, Any],
) -> None:
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
            st.session_state.pop(
                f"learning_dataset_info_open_{dataset['id']}",
                None,
            )
            _notify("learning_dataset_notice", "数据集信息已保存")
            _rerun()
        except Exception as exc:
            _show_error(exc)
    if st.button(
        "取消",
        key=f"learning_dataset_info_cancel_{dataset['id']}",
        width="stretch",
    ):
        st.session_state.pop(
            f"learning_dataset_info_open_{dataset['id']}",
            None,
        )
        st.rerun()


def _render_dataset_management(
    service: LearningDataService,
    dataset: dict[str, Any],
) -> None:
    info_key = f"learning_dataset_info_open_{dataset['id']}"
    if st.button(
        "数据集信息",
        icon=":material/info:",
        type="tertiary",
        key=f"learning_dataset_info_toggle_{dataset['id']}",
    ):
        st.session_state[info_key] = True
        st.rerun()
    if st.session_state.get(info_key):
        _dataset_info_dialog(service, dataset)


def _render_dataset_create_form(service: LearningDataService) -> None:
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
            width="stretch",
        )
    if submitted:
        try:
            dataset = service.create_dataset(name, description)
            st.session_state["learning_dataset_selector"] = dataset["id"]
            _notify("learning_dataset_notice", "学情数据集已创建")
            _rerun()
        except Exception as exc:
            _show_error(exc)


def render_dataset_context(
    service: LearningDataService,
    *,
    mode: str = "edit",
) -> str | None:
    """Render dataset selection; creating datasets is only available in import."""
    _show_notice("learning_dataset_notice")
    datasets = service.list_datasets()
    dataset_ids = [row["id"] for row in datasets]
    selector_key = "learning_dataset_selector"
    selected_id = st.session_state.get(selector_key)
    if selected_id not in dataset_ids:
        selected_id = None
        st.session_state[selector_key] = None
    allow_create = mode == "import"

    if selected_id is None:
        if allow_create:
            st.markdown("**新建学情数据集**")
            _render_dataset_create_form(service)
            st.caption(
                "创建后会自动打开该数据集；向已有数据集导入时，"
                "请先在“浏览”或“编辑”栏目打开该数据集。"
            )
            return None
        with st.container(
            horizontal=True,
            vertical_alignment="center",
            key="learning_dataset_list_header",
        ):
            st.markdown("**学情数据集**")
        if not datasets:
            st.info(
                "尚无学情数据集，请切换到“导入和导出”栏目新建数据集。"
            )
            return None
        with st.container(key="material_browser_files"):
            with st.grid("auto", min_column_width=240, gap="small"):
                for dataset in datasets:
                    counts = {
                        spec.label: len(
                            service.list_records(dataset["id"], spec.key)
                        )
                        for spec in DATA_TYPE_SPECS.values()
                    }
                    with st.container(
                        border=True,
                        key=f"learning_dataset_card_{dataset['id']}",
                    ):
                        st.markdown(f"**{dataset['name']}**")
                        if dataset.get("description"):
                            st.caption(dataset["description"])
                        st.caption(
                            f"{sum(counts.values())} 条记录 · "
                            f"学生 {counts['模拟学生']} · "
                            f"作答 {counts['作答记录']}"
                        )
                        if st.button(
                            "查看",
                            key=f"learning_dataset_open_{dataset['id']}",
                            width="stretch",
                        ):
                            st.session_state[selector_key] = dataset["id"]
                            st.rerun()
        return None

    dataset = next(row for row in datasets if row["id"] == selected_id)
    with st.container(
        horizontal=True,
        vertical_alignment="center",
        key="learning_dataset_path",
    ):
        st.markdown("材料中心 /")
        if st.button(
            "学情数据集",
            type="tertiary",
            key="learning_dataset_path_root",
        ):
            st.session_state[selector_key] = None
            st.rerun()
        st.markdown("/")
        st.markdown(f"**{dataset['name']}**")
        if mode == "edit":
            _render_dataset_management(service, dataset)
    if allow_create:
        st.markdown("**新建学情数据集**")
        _render_dataset_create_form(service)
    return str(selected_id)


def render_data_type_page(
    service: LearningDataService,
    dataset_id: str,
    data_type: str,
    *,
    mode: str,
) -> None:
    spec = DATA_TYPE_SPECS[data_type]
    _render_type(service, dataset_id, spec, mode=mode)


def render_learning_data_page(service: LearningDataService) -> None:
    dataset_id = render_dataset_context(service)
    if dataset_id is None:
        return
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
            _render_type(service, dataset_id, spec, mode="edit")
    with tabs[-1]:
        _render_quality(service, dataset_id)
