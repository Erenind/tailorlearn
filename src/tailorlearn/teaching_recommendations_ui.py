from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf
import streamlit as st

from tailorlearn.ai import AIRequestError
from tailorlearn.learning_data import LearningDataService
from tailorlearn.models import ProviderProfile
from tailorlearn.teaching_chat import (
    OpenAICompatibleAssistantClient,
    TeachingAssistantService,
)


TEACHING_CHAT_UI_VERSION = "2026-11-13-teaching-chat-ui-v1"
HISTORY_PAGE_SIZE = 20
TEMPORARY_FILE_CHAR_LIMIT = 60_000
PURE_MODE_KEY = "teaching_chat_pure_mode"
DATASET_KEY = "teaching_chat_dataset"
CHAT_INPUT_KEY = "teaching_chat_input"
COMMON_QUESTION_KEY = "teaching_chat_common_question"
COMMON_QUESTION_EDIT_KEY = "teaching_chat_question_edit"
HISTORY_LIMIT_KEY = "teaching_chat_history_limit"
PLACEHOLDER_INDEX_KEY = "teaching_chat_placeholder_index"
TEMP_FILES_KEY = "teaching_chat_temporary_files"
TEMP_IGNORED_FILES_KEY = "teaching_chat_ignored_temporary_files"
PENDING_ACTION_KEY = "teaching_chat_pending_action"
QUESTION_NOTICE_KEY = "teaching_chat_question_notice"


def _temporary_file_text(uploaded_file: Any) -> tuple[str, str]:
    suffix = Path(str(uploaded_file.name)).suffix.casefold()
    data = uploaded_file.getvalue()
    if suffix == ".pdf":
        try:
            with pymupdf.open(stream=data, filetype="pdf") as document:
                text = "\n\n".join(
                    f"【第 {index} 页】\n{page.get_text('text')}"
                    for index, page in enumerate(document, start=1)
                ).strip()
        except Exception as exc:
            raise ValueError(f"{uploaded_file.name} 无法读取：{exc}") from exc
        if not text:
            raise ValueError(
                f"{uploaded_file.name} 没有可提取文字，暂不支持扫描版 PDF"
            )
        return text[:TEMPORARY_FILE_CHAR_LIMIT], "pdf"
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{uploaded_file.name} 不是 UTF-8 文本，无法读取"
        ) from exc
    return text[:TEMPORARY_FILE_CHAR_LIMIT], suffix.lstrip(".") or "text"


def _merge_temporary_files(service_files: list[dict[str, Any]]) -> None:
    existing = st.session_state.setdefault(TEMP_FILES_KEY, [])
    ignored = set(st.session_state.get(TEMP_IGNORED_FILES_KEY, []))
    known = {str(item["signature"]) for item in existing}
    errors: list[str] = []
    for uploaded in service_files:
        signature = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if signature in known or signature in ignored:
            continue
        try:
            text, file_type = _temporary_file_text(uploaded)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        existing.append(
            {
                "signature": signature,
                "name": str(uploaded.name),
                "type": file_type,
                "text": text,
            }
        )
        known.add(signature)
    for error in errors:
        st.error(error)


def _dataset_options(learning: LearningDataService) -> list[dict[str, Any]]:
    return learning.list_datasets()


def _fill_common_question(
    suggestions: list[dict[str, Any]],
) -> None:
    suggestion_id = st.session_state.get(COMMON_QUESTION_KEY)
    selected = next(
        (item for item in suggestions if item["id"] == suggestion_id),
        None,
    )
    if selected is None:
        return
    st.session_state[CHAT_INPUT_KEY] = selected["text"]
    st.session_state[PLACEHOLDER_INDEX_KEY] = (
        int(st.session_state.get(PLACEHOLDER_INDEX_KEY, 0)) + 1
    )
    st.session_state[COMMON_QUESTION_KEY] = "__choose__"


def _render_question_manager(service: TeachingAssistantService) -> None:
    suggestions = service.list_suggestions()
    notice = st.session_state.pop(QUESTION_NOTICE_KEY, None)
    if notice:
        st.success(notice)

    with st.form("teaching_question_add", clear_on_submit=True):
        new_question = st.text_input(
            "新增常用问题",
            placeholder="输入一个问题",
            key="teaching_question_new",
        )
        add = st.form_submit_button(
            "添加",
            icon=":material/add:",
            width="stretch",
        )
    if add:
        try:
            service.add_suggestion(new_question)
            st.session_state[QUESTION_NOTICE_KEY] = "常用问题已添加"
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    if not suggestions:
        return
    edit_options = [item["id"] for item in suggestions]
    if st.session_state.get(COMMON_QUESTION_EDIT_KEY) not in edit_options:
        st.session_state[COMMON_QUESTION_EDIT_KEY] = edit_options[0]
    selected_id = st.selectbox(
        "编辑常用问题",
        edit_options,
        key=COMMON_QUESTION_EDIT_KEY,
        format_func=lambda value: next(
            item["text"] for item in suggestions if item["id"] == value
        ),
    )
    selected = next(
        item for item in suggestions if item["id"] == selected_id
    )
    edited_text = st.text_area(
        "问题内容",
        value=selected["text"],
        key=f"teaching_question_edit_text_{selected_id}",
        height=82,
    )
    save_column, delete_column = st.columns(2)
    if save_column.button(
        "保存修改",
        icon=":material/check:",
        key=f"teaching_question_save_{selected_id}",
        width="stretch",
    ):
        try:
            service.update_suggestion(selected_id, edited_text)
            st.session_state[QUESTION_NOTICE_KEY] = "常用问题已更新"
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if delete_column.button(
        "删除",
        icon=":material/delete:",
        key=f"teaching_question_delete_{selected_id}",
        width="stretch",
    ):
        try:
            service.delete_suggestion(selected_id)
            st.session_state[QUESTION_NOTICE_KEY] = "常用问题已删除"
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def _render_material_uploader() -> None:
    files = st.file_uploader(
        "上传临时素材",
        type=["pdf", "txt", "md", "csv", "json"],
        accept_multiple_files=True,
        key="teaching_temporary_upload",
    )
    if files:
        _merge_temporary_files(files)
    temporary_files = st.session_state.get(TEMP_FILES_KEY, [])
    if temporary_files:
        st.caption(
            "已附加："
            + "、".join(str(item["name"]) for item in temporary_files)
        )
        if st.button(
            "清除临时素材",
            icon=":material/delete_sweep:",
            key="teaching_temporary_clear",
            width="stretch",
        ):
            st.session_state[TEMP_IGNORED_FILES_KEY] = [
                str(item["signature"]) for item in temporary_files
            ]
            st.session_state[TEMP_FILES_KEY] = []
            st.rerun()
    st.caption("临时素材只用于当前对话，不会写入数据库。")


def _render_pending_action(
    service: TeachingAssistantService,
    profile: ProviderProfile | None,
) -> None:
    action = st.session_state.get(PENDING_ACTION_KEY)
    if not isinstance(action, dict):
        return
    with st.container(border=True, key="teaching_pending_action"):
        st.subheader("等待确认", icon=":material/fact_check:")
        st.write(action.get("summary") or "即将执行一项写操作。")
        changes = action.get("changes")
        if isinstance(changes, dict) and changes:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "字段": key,
                            "原值": value.get("before"),
                            "新值": value.get("after"),
                        }
                        for key, value in changes.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
                alt="待确认学情数据修改差异",
            )
        confirm, cancel = st.columns(2)
        if confirm.button(
            "确认执行",
            icon=":material/check_circle:",
            type="primary",
            key="teaching_pending_confirm",
            width="stretch",
        ):
            if profile is None or not profile.api_key:
                st.error("请先启用 AI 接入配置")
                return
            try:
                with st.spinner("正在执行已确认操作..."):
                    result = service.execute_pending_action(action, profile)
                if result.get("status") == "created":
                    message = (
                        f"已创建学习内容清单《{result['run_name']}》，"
                        f"共 {result['item_count']} 条学习内容、"
                        f"{result['relation_count']} 条联系。"
                    )
                else:
                    message = (
                        f"已写入 {result['data_type']} 数据，"
                        f"记录 row_id={result['row_id']}。"
                    )
                service.append_message("assistant", message)
                st.session_state.pop(PENDING_ACTION_KEY, None)
                st.rerun()
            except (AIRequestError, ValueError) as exc:
                st.error(str(exc))
        if cancel.button(
            "取消",
            key="teaching_pending_cancel",
            width="stretch",
        ):
            service.append_message("assistant", "已取消待确认操作。")
            st.session_state.pop(PENDING_ACTION_KEY, None)
            st.rerun()


def _render_messages(service: TeachingAssistantService) -> None:
    total = service.count_messages()
    history_limit = max(
        HISTORY_PAGE_SIZE,
        int(st.session_state.get(HISTORY_LIMIT_KEY, HISTORY_PAGE_SIZE)),
    )
    if total > history_limit:
        if st.button(
            "加载更早对话",
            icon=":material/history:",
            key="teaching_load_earlier",
        ):
            st.session_state[HISTORY_LIMIT_KEY] = history_limit + HISTORY_PAGE_SIZE
            st.rerun()
    messages = service.list_messages(history_limit)
    with st.container(
        height=500,
        border=False,
        autoscroll=True,
        key="teaching_chat_messages",
    ):
        if not messages:
            st.info(
                "可以直接提问，也可以从下方选择常用问题。"
                "AI 会按需读取课本目录、正文和已授权的数据。"
            )
        for message in messages:
            with st.chat_message(
                message["role"],
                avatar=":material/robot:" if message["role"] == "assistant" else None,
            ):
                st.markdown(message["content"])


def _render_composer(
    service: TeachingAssistantService,
    learning: LearningDataService,
    profile: ProviderProfile | None,
) -> str | None:
    suggestions = service.list_suggestions()
    datasets = _dataset_options(learning)
    placeholder_index = int(st.session_state.get(PLACEHOLDER_INDEX_KEY, 0))
    if suggestions:
        showcase = suggestions[placeholder_index % len(suggestions)]["text"]
        placeholder = f"从下方选择并编辑 · {showcase}"
    else:
        placeholder = "从下方选择并编辑问题"

    prompt: str | None = None
    chat_available = profile is not None and bool(profile.api_key)
    with st.bottom:
        with st.container(
            border=True,
            key="teaching_chat_composer",
        ):
            with st.container(
                horizontal=True,
                vertical_alignment="center",
                wrap=True,
                gap="small",
                key="teaching_chat_toolbar",
            ):
                pure_mode = st.toggle(
                    "纯净模式",
                    key=PURE_MODE_KEY,
                    help="不读取学生、班级、作答和学习主题数据",
                )
                dataset_id: str | None = None
                if not pure_mode:
                    if datasets:
                        dataset_options = [item["id"] for item in datasets]
                        if st.session_state.get(DATASET_KEY) not in dataset_options:
                            st.session_state[DATASET_KEY] = dataset_options[0]
                        dataset_id = st.selectbox(
                            "学情数据集",
                            dataset_options,
                            key=DATASET_KEY,
                            format_func=lambda value: next(
                                item["name"]
                                for item in datasets
                                if item["id"] == value
                            ),
                            persist_state="page",
                        )
                    else:
                        st.caption("尚无学情数据集")
                with st.popover(
                    "临时素材",
                    icon=":material/add:",
                ):
                    _render_material_uploader()
                if suggestions:
                    question_options = [
                        "__choose__",
                        *[item["id"] for item in suggestions],
                    ]
                    if (
                        st.session_state.get(COMMON_QUESTION_KEY)
                        not in question_options
                    ):
                        st.session_state[COMMON_QUESTION_KEY] = "__choose__"
                    st.selectbox(
                        "常用问题",
                        question_options,
                        key=COMMON_QUESTION_KEY,
                        format_func=lambda value: (
                            "选择常用问题"
                            if value == "__choose__"
                            else next(
                                item["text"]
                                for item in suggestions
                                if item["id"] == value
                            )
                        ),
                        on_change=_fill_common_question,
                        args=(suggestions,),
                    )
                with st.popover(
                    "管理问题",
                    icon=":material/edit_note:",
                ):
                    _render_question_manager(service)
            prompt = st.chat_input(
                placeholder,
                key=CHAT_INPUT_KEY,
                submit_mode="disable",
                disabled=not chat_available,
            )
            if not chat_available:
                st.caption("请先在“AI 参数设置”中启用 AI 接入")
    if prompt:
        return str(prompt)
    return None


def render_teaching_recommendations_page(
    service: TeachingAssistantService,
    learning: LearningDataService,
    profile: ProviderProfile | None,
) -> None:
    st.session_state.setdefault(HISTORY_LIMIT_KEY, HISTORY_PAGE_SIZE)
    st.session_state.setdefault(TEMP_FILES_KEY, [])
    st.session_state.setdefault(TEMP_IGNORED_FILES_KEY, [])
    _render_messages(service)
    _render_pending_action(service, profile)
    turn_slot = st.container()
    prompt = _render_composer(service, learning, profile)
    if not prompt:
        return

    pure_mode = bool(st.session_state.get(PURE_MODE_KEY, False))
    dataset_id = None
    if not pure_mode:
        dataset_id = st.session_state.get(DATASET_KEY)
    temporary_files = list(st.session_state.get(TEMP_FILES_KEY, []))
    result = None
    with turn_slot:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=":material/robot:"):
            try:
                assert profile is not None
                with st.status(
                    "正在结合教学资料生成建议",
                    expanded=False,
                ) as status:
                    result = service.run_turn(
                        prompt=prompt,
                        profile=profile,
                        client=OpenAICompatibleAssistantClient(profile),
                        dataset_id=dataset_id if isinstance(dataset_id, str) else None,
                        pure_mode=pure_mode,
                        temporary_materials=temporary_files,
                    )
                    status.update(
                        label="教学建议已生成",
                        state="complete",
                    )
                st.markdown(result.message["content"])
            except (AIRequestError, ValueError) as exc:
                service.append_message("assistant", f"请求失败：{exc}")
                st.error(str(exc))
    if result is not None and result.pending_action:
        st.session_state[PENDING_ACTION_KEY] = result.pending_action
    st.session_state[PLACEHOLDER_INDEX_KEY] = (
        int(st.session_state.get(PLACEHOLDER_INDEX_KEY, 0)) + 1
    )
    st.session_state[HISTORY_LIMIT_KEY] = HISTORY_PAGE_SIZE
    st.rerun()
