from __future__ import annotations

import hashlib
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


TEACHING_CHAT_UI_VERSION = "2026-11-13-teaching-chat-ui-v3"
HISTORY_PAGE_SIZE = 20
TEMPORARY_FILE_CHAR_LIMIT = 60_000
PURE_MODE_KEY = "teaching_chat_pure_mode"
DATASET_KEY = "teaching_chat_dataset"
DRAFT_KEY = "teaching_chat_draft"
HISTORY_LIMIT_KEY = "teaching_chat_history_limit"
PLACEHOLDER_INDEX_KEY = "teaching_chat_placeholder_index"
TEMP_FILES_KEY = "teaching_chat_temporary_files"
PENDING_ACTION_KEY = "teaching_chat_pending_action"
QUESTION_ACTION_KEY = "teaching_chat_question_action"
RESET_DRAFT_KEY = "teaching_chat_reset_draft"
QUESTION_POPOVER_KEY = "teaching_question_popover"
DATASET_POPOVER_KEY = "teaching_dataset_popover"


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
    known = {str(item["signature"]) for item in existing}
    errors: list[str] = []
    for uploaded in service_files:
        signature = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if signature in known:
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


def _use_common_question(text: str) -> None:
    st.session_state[DRAFT_KEY] = text
    st.session_state[QUESTION_POPOVER_KEY] = False
    st.session_state[PLACEHOLDER_INDEX_KEY] = (
        int(st.session_state.get(PLACEHOLDER_INDEX_KEY, 0)) + 1
    )


@st.dialog("新增常用问题", width="small")
def _add_question_dialog(service: TeachingAssistantService) -> None:
    text = st.text_area(
        "问题内容",
        placeholder="输入要保存的问题",
        key="teaching_question_dialog_add_text",
        height=90,
    )
    save, cancel = st.columns(2)
    if save.button(
        "保存",
        icon=":material/check:",
        type="primary",
        key="teaching_question_dialog_add_save",
        width="stretch",
    ):
        try:
            service.add_suggestion(text)
            st.session_state.pop(QUESTION_ACTION_KEY, None)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if cancel.button(
        "取消",
        key="teaching_question_dialog_add_cancel",
        width="stretch",
    ):
        st.session_state.pop(QUESTION_ACTION_KEY, None)
        st.rerun()


@st.dialog("编辑常用问题", width="small")
def _edit_question_dialog(
    service: TeachingAssistantService,
    suggestion_id: str,
) -> None:
    suggestion = next(
        (
            item
            for item in service.list_suggestions()
            if item["id"] == suggestion_id
        ),
        None,
    )
    if suggestion is None:
        st.session_state.pop(QUESTION_ACTION_KEY, None)
        st.rerun()
    text = st.text_area(
        "问题内容",
        value=suggestion["text"],
        key=f"teaching_question_dialog_edit_{suggestion_id}",
        height=90,
    )
    save, cancel = st.columns(2)
    if save.button(
        "保存修改",
        icon=":material/check:",
        type="primary",
        key=f"teaching_question_dialog_save_{suggestion_id}",
        width="stretch",
    ):
        try:
            service.update_suggestion(suggestion_id, text)
            st.session_state.pop(QUESTION_ACTION_KEY, None)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if cancel.button(
        "取消",
        key=f"teaching_question_dialog_cancel_{suggestion_id}",
        width="stretch",
    ):
        st.session_state.pop(QUESTION_ACTION_KEY, None)
        st.rerun()


@st.dialog("删除常用问题", width="small")
def _delete_question_dialog(
    service: TeachingAssistantService,
    suggestion_id: str,
) -> None:
    suggestion = next(
        (
            item
            for item in service.list_suggestions()
            if item["id"] == suggestion_id
        ),
        None,
    )
    if suggestion is None:
        st.session_state.pop(QUESTION_ACTION_KEY, None)
        st.rerun()
    st.warning(f"确定删除“{suggestion['text']}”吗？")
    confirm, cancel = st.columns(2)
    if confirm.button(
        "删除",
        icon=":material/delete:",
        type="primary",
        key=f"teaching_question_dialog_delete_{suggestion_id}",
        width="stretch",
    ):
        service.delete_suggestion(suggestion_id)
        st.session_state.pop(QUESTION_ACTION_KEY, None)
        st.rerun()
    if cancel.button(
        "取消",
        key=f"teaching_question_dialog_delete_cancel_{suggestion_id}",
        width="stretch",
    ):
        st.session_state.pop(QUESTION_ACTION_KEY, None)
        st.rerun()


def _render_question_dialog(service: TeachingAssistantService) -> None:
    pending = st.session_state.get(QUESTION_ACTION_KEY)
    if not isinstance(pending, dict):
        return
    action = pending.get("action")
    suggestion_id = str(pending.get("suggestion_id") or "")
    if action == "add":
        _add_question_dialog(service)
    elif action == "edit" and suggestion_id:
        _edit_question_dialog(service, suggestion_id)
    elif action == "delete" and suggestion_id:
        _delete_question_dialog(service, suggestion_id)
    else:
        st.session_state.pop(QUESTION_ACTION_KEY, None)


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


def _apply_composer_style() -> None:
    st.html(
        """
        <style>
        .st-key-teaching_chat_composer {
            padding: 0.1rem 0.15rem;
            border: 1px solid #D8DEE8 !important;
            border-radius: 14px !important;
            background: #FFFFFF !important;
        }
        .st-key-teaching_chat_input_area {
            padding: 0.45rem 0.65rem 0.15rem;
            border-bottom: 1px solid #E1E5EB;
        }
        .st-key-teaching_chat_draft,
        .st-key-teaching_chat_draft > div,
        .st-key-teaching_chat_draft > div > div {
            width: 100%;
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        .st-key-teaching_chat_draft textarea {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
            padding-left: 0.05rem;
        }
        .st-key-teaching_chat_toolbar {
            padding: 0.25rem 0.3rem 0.1rem;
        }
        .st-key-teaching_chat_pure_mode button {
            min-height: 2rem;
            border: 1px solid #D0D5DD !important;
            border-radius: 999px !important;
            background: transparent !important;
            color: #667085 !important;
            box-shadow: none !important;
        }
        .st-key-teaching_chat_pure_mode button[data-selected] {
            border-color: #2F5BEA !important;
            background: #EDF2FF !important;
            color: #2F5BEA !important;
        }
        .st-key-teaching_chat_pure_mode button[data-selected] * {
            color: #2F5BEA !important;
            fill: #2F5BEA !important;
        }
        .st-key-teaching_chat_send button {
            width: 2.55rem;
            min-width: 2.55rem;
            height: 2.55rem;
            padding: 0;
            border-radius: 999px !important;
        }
        .st-key-teaching_chat_send button p {
            display: none;
        }
        [class*="st-key-teaching_question_edit_"] button,
        [class*="st-key-teaching_question_delete_"] button {
            min-height: 2rem;
            width: 2rem;
            padding: 0.15rem;
        }
        [class*="st-key-teaching_question_edit_"] button p,
        [class*="st-key-teaching_question_delete_"] button p {
            display: none;
        }
        </style>
        """
    )


def _select_pure_mode() -> tuple[Any, bool]:
    state = st.session_state.get(PURE_MODE_KEY)
    if isinstance(state, bool):
        st.session_state[PURE_MODE_KEY] = "纯净模式" if state else None
    selected = st.pills(
        "模式",
        ["纯净模式"],
        default=None,
        format_func=lambda _: ":material/eco: 纯净模式",
        key=PURE_MODE_KEY,
        label_visibility="collapsed",
        width="content",
    )
    return selected, selected == "纯净模式"


def _select_dataset(dataset_id: str) -> None:
    st.session_state[DATASET_KEY] = dataset_id
    st.session_state[DATASET_POPOVER_KEY] = False


def _queue_question_action(action: str, suggestion_id: str = "") -> None:
    st.session_state[QUESTION_POPOVER_KEY] = False
    st.session_state[QUESTION_ACTION_KEY] = {
        "action": action,
        "suggestion_id": suggestion_id,
    }


def _render_dataset_popover(
    datasets: list[dict[str, Any]],
) -> None:
    dataset_ids = [str(item["id"]) for item in datasets]
    current_id = st.session_state.get(DATASET_KEY)
    if current_id not in dataset_ids and dataset_ids:
        current_id = dataset_ids[0]
        st.session_state[DATASET_KEY] = current_id
    current_name = next(
        (
            str(item["name"])
            for item in datasets
            if str(item["id"]) == current_id
        ),
        "选择数据集",
    )
    with st.popover(
        current_name,
        icon=":material/database:",
        key=DATASET_POPOVER_KEY,
        width="content",
        on_change="rerun",
    ):
        if not datasets:
            st.caption("尚无学情数据集")
            return
        for item in datasets:
            selected = str(item["id"]) == str(current_id)
            st.button(
                str(item["name"]) + (" · 当前" if selected else ""),
                key=f"teaching_dataset_choose_{item['id']}",
                type="tertiary",
                width="stretch",
                on_click=_select_dataset,
                args=(str(item["id"]),),
            )


def _render_question_popover(
    suggestions: list[dict[str, Any]],
) -> None:
    with st.popover(
        "常用问题",
        icon=":material/quiz:",
        key=QUESTION_POPOVER_KEY,
        width="content",
        on_change="rerun",
    ):
        with st.container(height=320, gap="xsmall"):
            for item in suggestions:
                with st.container(
                    horizontal=True,
                    vertical_alignment="center",
                    gap="xsmall",
                    key=f"teaching_question_row_{item['id']}",
                ):
                    st.button(
                        str(item["text"]),
                        key=f"teaching_question_use_{item['id']}",
                        type="tertiary",
                        width="stretch",
                        on_click=_use_common_question,
                        args=(str(item["text"]),),
                    )
                    st.badge(
                        "预设" if item["is_preset"] else "自定义",
                        color="violet" if item["is_preset"] else "blue",
                    )
                    st.button(
                        "编辑",
                        icon=":material/edit:",
                        key=f"teaching_question_edit_{item['id']}",
                        type="tertiary",
                        help="编辑该问题",
                        on_click=_queue_question_action,
                        args=("edit", str(item["id"])),
                    )
                    st.button(
                        "删除",
                        icon=":material/delete:",
                        key=f"teaching_question_delete_{item['id']}",
                        type="tertiary",
                        help="删除该问题",
                        on_click=_queue_question_action,
                        args=("delete", str(item["id"])),
                    )
        st.button(
            "新增问题",
            icon=":material/add:",
            key="teaching_question_add_open",
            type="primary",
            width="stretch",
            on_click=_queue_question_action,
            args=("add",),
        )


def _render_attachment_popover() -> None:
    with st.popover(
        ":material/attach_file:",
        help="添加临时素材",
        key="teaching_attachment_popover",
        width="content",
    ):
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
                st.session_state[TEMP_FILES_KEY] = []
                st.rerun()
        st.caption("仅用于当前对话，不写入数据库。")


def _render_composer(
    service: TeachingAssistantService,
    learning: LearningDataService,
    profile: ProviderProfile | None,
) -> str | None:
    _apply_composer_style()
    suggestions = service.list_suggestions()
    datasets = _dataset_options(learning)
    if st.session_state.pop(RESET_DRAFT_KEY, False):
        st.session_state[DRAFT_KEY] = ""
    st.session_state.setdefault(DRAFT_KEY, "")
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
            with st.container(key="teaching_chat_input_area"):
                st.text_area(
                    "问题",
                    key=DRAFT_KEY,
                    placeholder=placeholder,
                    height=92,
                    label_visibility="collapsed",
                )
            with st.container(
                key="teaching_chat_toolbar",
            ):
                left_tools, right_actions = st.columns(
                    [6.5, 1.0],
                    vertical_alignment="center",
                    gap="small",
                )
                with left_tools:
                    with st.container(
                        horizontal=True,
                        vertical_alignment="center",
                        gap="xsmall",
                    ):
                        _, pure_mode = _select_pure_mode()
                        if not pure_mode:
                            _render_dataset_popover(datasets)
                        _render_question_popover(suggestions)
                with right_actions:
                    with st.container(
                        horizontal=True,
                        vertical_alignment="center",
                        horizontal_alignment="right",
                        gap="xsmall",
                    ):
                        _render_attachment_popover()
                        send_clicked = st.button(
                            "发送",
                            icon=":material/send:",
                            key="teaching_chat_send",
                            type="primary",
                            help="发送",
                            disabled=not chat_available,
                        )
            if not chat_available:
                st.caption("请先在“AI 参数设置”中启用 AI 接入")
            if send_clicked:
                draft = str(st.session_state.get(DRAFT_KEY) or "").strip()
                if not draft and not st.session_state.get(TEMP_FILES_KEY):
                    st.toast("请输入问题或添加临时素材")
                else:
                    prompt = draft or (
                        "请阅读我已附加的临时素材，并结合当前对话继续分析。"
                    )
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
    _render_question_dialog(service)
    _render_messages(service)
    _render_pending_action(service, profile)
    turn_slot = st.container()
    prompt = _render_composer(service, learning, profile)
    if not prompt:
        return

    pure_mode = st.session_state.get(PURE_MODE_KEY) == "纯净模式"
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
    st.session_state[RESET_DRAFT_KEY] = True
    st.session_state[HISTORY_LIMIT_KEY] = HISTORY_PAGE_SIZE
    st.rerun()
