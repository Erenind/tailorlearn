from __future__ import annotations

import base64
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
    TEACHING_ACTION_PROMPTS,
    OpenAICompatibleAssistantClient,
    TeachingContentDelta,
    TeachingAssistantService,
    TeachingToolFinished,
    TeachingToolStarted,
    TeachingTurnComplete,
    TeachingTurnResult,
)


TEACHING_CHAT_UI_VERSION = "2026-11-14-teaching-chat-ui-v16"
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
RETRY_PROMPT_KEY = "teaching_chat_retry_prompt"
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


def _suggestion_prompt(suggestion_id: str, text: str) -> str:
    """Expand a preset suggestion into its full teaching prompt."""
    return TEACHING_ACTION_PROMPTS.get(suggestion_id, text)


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


def _message_turns(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    turns: list[list[dict[str, Any]]] = []
    for message in messages:
        if message["role"] == "user" or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def _render_messages(
    service: TeachingAssistantService,
    profile: ProviderProfile | None,
) -> tuple[dict[str, str], Any]:
    copy_payload: dict[str, str] = {}
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
        height="stretch",
        border=False,
        autoscroll=True,
        key="teaching_chat_messages",
    ):
        if not messages:
            st.info(
                "可以直接提问，也可以从下方选择常用问题。"
                "AI 会按需读取课本目录、正文和已授权的数据。"
            )
        for turn in _message_turns(messages):
            user_message = next(
                (
                    message
                    for message in turn
                    if message["role"] == "user"
                ),
                None,
            )
            turn_key = str(
                user_message["id"] if user_message is not None else turn[0]["id"]
            )
            with st.container(
                gap="xsmall",
                key=f"teaching_chat_turn_{turn_key}",
            ):
                assistant_message = next(
                    (
                        message
                        for message in reversed(turn)
                        if message["role"] == "assistant"
                    ),
                    None,
                )
                assistant_content = (
                    str(assistant_message["content"])
                    if assistant_message is not None
                    else ""
                )
                if user_message is not None and assistant_content:
                    copy_payload[turn_key] = assistant_content
                for message in turn:
                    with st.chat_message(
                        message["role"],
                        avatar=(
                            ":material/robot:"
                            if message["role"] == "assistant"
                            else None
                        ),
                    ):
                        tool_trace = message.get("tool_trace")
                        if (
                            message["role"] == "assistant"
                            and isinstance(tool_trace, list)
                            and tool_trace
                        ):
                            failed = sum(
                                item.get("status") == "error"
                                for item in tool_trace
                                if isinstance(item, dict)
                            )
                            label = (
                                f"资料调用完成，{failed} 项失败"
                                if failed
                                else f"已完成 {len(tool_trace)} 项资料调用"
                            )
                            with st.status(
                                label,
                                expanded=False,
                                state="error" if failed else "complete",
                                type="compact",
                            ):
                                for item in tool_trace:
                                    if not isinstance(item, dict):
                                        continue
                                    item_label = str(
                                        item.get("label")
                                        or item.get("tool_name")
                                        or "调用工具"
                                    )
                                    icon = (
                                        ":material/error:"
                                        if item.get("status") == "error"
                                        else ":material/check_circle:"
                                    )
                                    st.markdown(f"{icon} {item_label}")
                                    trace_detail = (
                                        item.get("detail")
                                        or item.get("progress")
                                    )
                                    if trace_detail:
                                        st.caption(str(trace_detail))
                        st.markdown(message["content"])
                        if (
                            message is assistant_message
                            and user_message is not None
                        ):
                            with st.container(
                                horizontal=True,
                                horizontal_alignment="left",
                                vertical_alignment="center",
                                gap="xxsmall",
                                key=f"teaching_chat_actions_{turn_key}",
                            ):
                                st.button(
                                    "复制",
                                    icon=":material/content_copy:",
                                    key=f"teaching_chat_copy_turn_{turn_key}",
                                    type="tertiary",
                                    help="复制结果",
                                )
                                if st.button(
                                    "重试",
                                    icon=":material/refresh:",
                                    key=f"teaching_chat_retry_turn_{turn_key}",
                                    type="tertiary",
                                    help="使用原问题重新生成结果",
                                    disabled=(
                                        profile is None or not profile.api_key
                                    ),
                                ):
                                    st.session_state[RETRY_PROMPT_KEY] = str(
                                        user_message["content"]
                                    )
                                    st.rerun()
                                if st.button(
                                    "删除",
                                    icon=":material/delete_outline:",
                                    key=f"teaching_chat_delete_turn_{turn_key}",
                                    type="tertiary",
                                    help="删除此问题及对应回复，无法恢复",
                                ):
                                    service.delete_conversation_turn(
                                        int(user_message["id"])
                                    )
                                    st.session_state.pop(
                                        PENDING_ACTION_KEY,
                                        None,
                                    )
                                    st.session_state[
                                        HISTORY_LIMIT_KEY
                                    ] = HISTORY_PAGE_SIZE
                                    st.session_state[RESET_DRAFT_KEY] = True
                                    st.toast("已删除此次对话")
                                    st.rerun()
        live_turn_slot = st.container(key="teaching_live_turn")
    return copy_payload, live_turn_slot


def _apply_composer_style(copy_payload: dict[str, str]) -> None:
    encoded_payload = {
        turn_key: base64.b64encode(content.encode("utf-8")).decode("ascii")
        for turn_key, content in copy_payload.items()
    }
    st.html(
        """
        <style>
        #Bottom > div:first-child {
            display: none !important;
        }
        .stMainBlockContainer:has(.st-key-teaching_page_shell) {
            display: flex;
            flex-direction: column;
            height: auto;
            min-height: 0;
            overflow: hidden;
            padding-bottom: 0.75rem !important;
        }
        .stMainBlockContainer:has(.st-key-teaching_page_shell)
        > [data-testid="stVerticalBlock"] {
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
            min-height: 0;
            overflow: hidden;
        }
        .stMainBlockContainer:has(.st-key-teaching_page_shell)
        > [data-testid="stVerticalBlock"]
        > [data-testid="stLayoutWrapper"]:has(.st-key-teaching_page_shell) {
            flex: 1 1 auto;
            display: flex;
            min-height: 0;
            overflow: hidden;
        }
        .st-key-teaching_page_shell {
            display: flex;
            flex: 1 1 auto;
            flex-direction: column;
            gap: 0.35rem;
            min-height: 0;
            overflow: hidden;
        }
        .st-key-teaching_page_shell
        > [data-testid="stLayoutWrapper"]:has(.st-key-teaching_chat_messages) {
            flex: 1 1 auto;
            display: flex;
            min-height: 0;
            overflow: hidden;
        }
        .st-key-teaching_chat_messages {
            flex: 1 1 auto;
            height: 100% !important;
            min-height: 0;
            overflow-x: hidden !important;
            overflow-y: auto !important;
        }
        [data-testid="stBottom"] {
            background: #FFFFFF !important;
        }
        [data-testid="stBottomBlockContainer"] {
            padding-top: 0 !important;
            padding-bottom: 0.75rem !important;
        }
        .st-key-teaching_chat_composer {
            position: relative;
            gap: 0;
        }
        .st-key-teaching_chat_draft [data-testid="stChatInput"] > div {
            border-width: 2px !important;
            border-radius: 1.8rem !important;
            min-height: 5.6rem;
            padding: 0.9rem 1rem 3.1rem !important;
        }
        .st-key-teaching_chat_draft [data-testid="stChatInputTextArea"] {
            font-size: 16px !important;
            line-height: 1.6 !important;
        }
        .st-key-teaching_chat_draft [data-testid="stChatInputSubmitButton"] {
            position: absolute !important;
            right: 0.9rem !important;
            bottom: 0.65rem !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 2.9rem !important;
            height: 2.9rem !important;
            padding: 0 !important;
            border-radius: 999px !important;
            z-index: 5 !important;
        }
        .st-key-teaching_chat_draft
        [data-testid="stChatInputFileUploadButton"] {
            position: absolute !important;
            right: 4.2rem !important;
            bottom: 0.65rem !important;
            z-index: 5 !important;
        }
        .st-key-teaching_chat_draft
        [data-testid="stChatInputFileUploadButton"] button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 2.55rem !important;
            height: 2.55rem !important;
            padding: 0 !important;
            border: none !important;
            border-radius: 999px !important;
            background: transparent !important;
        }
        .st-key-teaching_chat_draft
        [data-testid="stChatInputFileUploadButton"] svg {
            width: 1.35rem !important;
            height: 1.35rem !important;
        }
        .st-key-teaching_chat_draft
        [data-testid="stChatInputSubmitButton"]:not([disabled]) {
            background: #2F5BEA !important;
        }
        .st-key-teaching_chat_draft
        [data-testid="stChatInputSubmitButton"]:not([disabled]):hover {
            background: #2449C9 !important;
        }
        .st-key-teaching_chat_draft
        [data-testid="stChatInputSubmitButton"] svg {
            display: block !important;
            width: 1.42rem !important;
            height: 1.42rem !important;
            margin: auto !important;
        }
        .st-key-teaching_chat_draft
        [data-testid="stChatInputSubmitButton"] svg path {
            stroke: currentColor;
            stroke-width: 0.55;
            paint-order: stroke;
        }
        .st-key-teaching_chat_draft
        [data-testid="stChatInputSubmitButton"] svg path[fill="none"] {
            stroke: none !important;
        }
        .st-key-teaching_chat_toolbar {
            position: absolute;
            left: 1.05rem;
            bottom: 0.63rem;
            z-index: 4;
            width: auto;
            max-width: calc(100% - 9.5rem);
            padding: 0;
            flex-wrap: nowrap;
            overflow: visible;
        }
        .st-key-teaching_chat_toolbar [data-testid="stPopoverButton"] {
            min-height: 2rem;
            padding: 0.3rem 0.58rem;
        }
        .st-key-teaching_dataset_popover [data-testid="stPopoverButton"] {
            min-width: 5.25rem;
        }
        .st-key-teaching_question_popover [data-testid="stPopoverButton"] {
            min-width: 6.6rem;
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
        .st-key-teaching_dataset_popover
        [data-testid="stPopoverButton"] svg:last-of-type,
        .st-key-teaching_question_popover
        [data-testid="stPopoverButton"] svg:last-of-type {
            display: none !important;
        }
        [data-testid="stPopoverBody"] {
            padding: 0.45rem !important;
        }
        .st-key-teaching_dataset_list button,
        .st-key-teaching_question_list button {
            justify-content: flex-start !important;
            min-height: 1.9rem !important;
            padding: 0.2rem 0.42rem !important;
            text-align: left !important;
        }
        .st-key-teaching_dataset_list button > div,
        .st-key-teaching_question_list button > div,
        .st-key-teaching_dataset_list button p,
        .st-key-teaching_question_list button p {
            justify-content: flex-start !important;
            text-align: left;
            line-height: 1.35 !important;
        }
        .st-key-teaching_dataset_list button p,
        .st-key-teaching_question_list button p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        [class*="st-key-teaching_question_row_"] {
            min-height: 1.95rem;
            gap: 0.15rem !important;
        }
        [class*="st-key-teaching_question_edit_"] button,
        [class*="st-key-teaching_question_delete_"] button {
            min-height: 1.7rem;
            width: 1.7rem;
            padding: 0.1rem;
        }
        [class*="st-key-teaching_question_edit_"] button p,
        [class*="st-key-teaching_question_delete_"] button p {
            display: none;
        }
        [class*="st-key-teaching_chat_turn_"]
        [data-testid="stChatMessageAvatarUser"] {
            background: #2F5BEA !important;
            color: #FFFFFF !important;
        }
        [class*="st-key-teaching_chat_turn_"]
        [data-testid="stChatMessageAvatarUser"] svg {
            fill: #FFFFFF !important;
        }
        [class*="st-key-teaching_chat_copy_turn_"] button,
        [class*="st-key-teaching_chat_retry_turn_"] button,
        [class*="st-key-teaching_chat_delete_turn_"] button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 1.5rem !important;
            min-width: 1.5rem !important;
            min-height: 1.5rem !important;
            padding: 0 !important;
            border: none !important;
            border-radius: 0.4rem !important;
            background: transparent !important;
            color: #7A8496 !important;
            box-shadow: none !important;
        }
        [class*="st-key-teaching_chat_copy_turn_"] button svg,
        [class*="st-key-teaching_chat_retry_turn_"] button svg,
        [class*="st-key-teaching_chat_delete_turn_"] button svg {
            width: 1rem !important;
            height: 1rem !important;
        }
        [class*="st-key-teaching_chat_copy_turn_"] button p,
        [class*="st-key-teaching_chat_retry_turn_"] button p,
        [class*="st-key-teaching_chat_delete_turn_"] button p {
            display: none !important;
        }
        [class*="st-key-teaching_chat_copy_turn_"] button:hover,
        [class*="st-key-teaching_chat_retry_turn_"] button:hover {
            background: #F4F7FF !important;
            color: #2F5BEA !important;
        }
        [class*="st-key-teaching_chat_delete_turn_"] button:hover {
            background: #FFF1F2 !important;
            color: #C2415B !important;
        }
        </style>
        <script>
        (() => {
            const payload = __COPY_PAYLOAD__;
            const decode = (value) => new TextDecoder().decode(
                Uint8Array.from(atob(value), (char) => char.charCodeAt(0))
            );
            const fallbackCopy = (text) => {
                const textarea = document.createElement("textarea");
                textarea.value = text;
                textarea.style.position = "fixed";
                textarea.style.opacity = "0";
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand("copy");
                textarea.remove();
            };
            Object.entries(payload).forEach(([turnKey, encoded]) => {
                const button = document.querySelector(
                    `.st-key-teaching_chat_copy_turn_${turnKey} button`
                );
                if (!button || button.dataset.teachingCopyBound === "true") return;
                button.dataset.teachingCopyBound = "true";
                button.addEventListener("click", async () => {
                    const text = decode(encoded);
                    try {
                        await navigator.clipboard.writeText(text);
                    } catch (error) {
                        fallbackCopy(text);
                    }
                    button.title = "已复制";
                });
            });
            const messageBox = document.querySelector(
                ".st-key-teaching_chat_messages"
            );
            if (messageBox) {
                if (window.__teachingChatScrollObserver) {
                    window.__teachingChatScrollObserver.disconnect();
                }
                const autoFollowKey = "teachingChatAutoFollow";
                const scrollTopKey = "teachingChatScrollTop";
                const bottomThreshold = 24;
                let autoFollow = (
                    sessionStorage.getItem(autoFollowKey) !== "false"
                );
                const savedScrollTop = Number(
                    sessionStorage.getItem(scrollTopKey)
                );
                let anchoredLiveTurn = false;
                let scheduled = false;
                let suppressScrollEvent = false;
                const setScrollTop = (value) => {
                    const maxTop = Math.max(
                        0,
                        messageBox.scrollHeight - messageBox.clientHeight
                    );
                    suppressScrollEvent = true;
                    messageBox.scrollTop = Math.min(
                        Math.max(0, value),
                        maxTop
                    );
                    requestAnimationFrame(() => {
                        suppressScrollEvent = false;
                    });
                };
                const rememberScrollIntent = () => {
                    if (suppressScrollEvent) return;
                    const maxTop = Math.max(
                        0,
                        messageBox.scrollHeight - messageBox.clientHeight
                    );
                    autoFollow = (
                        maxTop - messageBox.scrollTop <= bottomThreshold
                    );
                    sessionStorage.setItem(
                        autoFollowKey,
                        autoFollow ? "true" : "false"
                    );
                    if (!autoFollow) {
                        sessionStorage.setItem(
                            scrollTopKey,
                            String(messageBox.scrollTop)
                        );
                    }
                };
                messageBox.addEventListener(
                    "scroll",
                    rememberScrollIntent,
                    { passive: true }
                );
                const scrollToLatest = () => {
                    if (!autoFollow) return;
                    const liveTurn = messageBox.querySelector(
                        ".st-key-teaching_live_turn"
                    );
                    const liveRect = liveTurn?.getBoundingClientRect();
                    const boxRect = messageBox.getBoundingClientRect();
                    if (liveRect && liveRect.height > 0 && !anchoredLiveTurn) {
                        setScrollTop(
                            messageBox.scrollTop + liveRect.top - boxRect.top
                        );
                        anchoredLiveTurn = true;
                        return;
                    }
                    setScrollTop(messageBox.scrollHeight);
                };
                const scheduleScroll = () => {
                    if (scheduled) return;
                    scheduled = true;
                    requestAnimationFrame(() => {
                        scheduled = false;
                        scrollToLatest();
                    });
                };
                const observer = new MutationObserver(scheduleScroll);
                const liveTurn = messageBox.querySelector(
                    ".st-key-teaching_live_turn"
                );
                if (liveTurn) {
                    observer.observe(liveTurn, {
                        childList: true,
                        subtree: true,
                        characterData: true,
                    });
                }
                window.__teachingChatScrollObserver = observer;
                const resumeFollowing = () => {
                    autoFollow = true;
                    sessionStorage.setItem(autoFollowKey, "true");
                };
                const submitButton = document.querySelector(
                    '[data-testid="stChatInputSubmitButton"]'
                );
                if (
                    submitButton
                    && submitButton.dataset.teachingScrollBound !== "true"
                ) {
                    submitButton.dataset.teachingScrollBound = "true";
                    submitButton.addEventListener("click", resumeFollowing);
                }
                const textInput = document.querySelector(
                    '[data-testid="stChatInputTextArea"]'
                );
                if (
                    textInput
                    && textInput.dataset.teachingScrollBound !== "true"
                ) {
                    textInput.dataset.teachingScrollBound = "true";
                    textInput.addEventListener("keydown", (event) => {
                        if (
                            event.key === "Enter"
                            && !event.shiftKey
                            && !event.isComposing
                        ) {
                            resumeFollowing();
                        }
                    });
                }
                const restoreInitialPosition = () => {
                    if (autoFollow) {
                        scrollToLatest();
                    } else if (Number.isFinite(savedScrollTop)) {
                        setScrollTop(savedScrollTop);
                    }
                };
                requestAnimationFrame(() => {
                    requestAnimationFrame(restoreInitialPosition);
                });
                setTimeout(restoreInitialPosition, 120);
            }
        })();
        </script>
        """.replace(
            "__COPY_PAYLOAD__",
            json.dumps(encoded_payload, ensure_ascii=True),
        ),
        unsafe_allow_javascript=True,
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
    with st.popover(
        "数据集",
        icon=":material/database:",
        key=DATASET_POPOVER_KEY,
        width="content",
        wrap=True,
        on_change="rerun",
    ):
        if not datasets:
            st.caption("尚无学情数据集")
            return
        list_height: int | str = 300 if len(datasets) > 8 else "content"
        with st.container(
            height=list_height,
            gap=0,
            key="teaching_dataset_list",
        ):
            for item in datasets:
                selected = str(item["id"]) == str(current_id)
                st.button(
                    str(item["name"]) + (" · 当前" if selected else ""),
                    key=f"teaching_dataset_choose_{item['id']}",
                    icon=(
                        ":material/check_circle:"
                        if selected
                        else ":material/database:"
                    ),
                    type="primary" if selected else "tertiary",
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
        wrap=True,
        on_change="rerun",
    ):
        list_height: int | str = 300 if len(suggestions) > 8 else "content"
        with st.container(
            height=list_height,
            gap=0,
            key="teaching_question_list",
        ):
            for item in suggestions:
                with st.container(
                    horizontal=True,
                    vertical_alignment="center",
                    gap="xxsmall",
                    key=f"teaching_question_row_{item['id']}",
                ):
                    st.button(
                        str(item["text"]),
                        key=f"teaching_question_use_{item['id']}",
                        type="tertiary",
                        width="stretch",
                        help=_suggestion_prompt(str(item["id"]), str(item["text"])),
                        on_click=_use_common_question,
                        args=(
                            _suggestion_prompt(str(item["id"]), str(item["text"])),
                        ),
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
            "新建常用问题",
            icon=":material/add:",
            key="teaching_question_add_open",
            type="primary",
            width="stretch",
            on_click=_queue_question_action,
            args=("add",),
        )


def _render_composer(
    service: TeachingAssistantService,
    learning: LearningDataService,
    profile: ProviderProfile | None,
    copy_payload: dict[str, str],
) -> str | None:
    _apply_composer_style(copy_payload)
    suggestions = service.list_suggestions()
    datasets = _dataset_options(learning)
    if st.session_state.pop(RESET_DRAFT_KEY, False):
        st.session_state[DRAFT_KEY] = ""
    st.session_state.setdefault(DRAFT_KEY, "")
    placeholder_index = int(st.session_state.get(PLACEHOLDER_INDEX_KEY, 0))
    if suggestions:
        showcase = suggestions[placeholder_index % len(suggestions)]["text"]
        placeholder = (
            "直接提问，公式用 $...$ 或 $$...$$，"
            f"也可以从下方选择常用问题，比如： {showcase}"
        )
    else:
        placeholder = (
            "直接提问，公式用 $...$ 或 $$...$$，"
            "也可以从下方选择常用问题"
        )

    chat_available = profile is not None and bool(profile.api_key)
    with st.bottom:
        with st.container(
            gap=0,
            key="teaching_chat_composer",
        ):
            with st.container(
                horizontal=True,
                vertical_alignment="center",
                gap="xsmall",
                wrap=False,
                key="teaching_chat_toolbar",
            ):
                _, pure_mode = _select_pure_mode()
                if not pure_mode:
                    _render_dataset_popover(datasets)
                _render_question_popover(suggestions)
                temporary_files = st.session_state.get(TEMP_FILES_KEY, [])
                if temporary_files:
                    if st.button(
                        "清除素材",
                        icon=":material/delete_sweep:",
                        key="teaching_temporary_clear",
                        type="tertiary",
                        help="清除当前对话附加的临时素材",
                    ):
                        st.session_state[TEMP_FILES_KEY] = []
                        st.rerun()
            submission = st.chat_input(
                placeholder,
                accept_file="multiple",
                file_type=["pdf", "txt", "md", "csv", "json"],
                key=DRAFT_KEY,
                disabled=not chat_available,
                height=92,
            )
            if not chat_available:
                st.caption("请先在“AI 参数设置”中启用 AI 接入")

    if submission is None:
        return None
    files = getattr(submission, "files", None)
    if files:
        _merge_temporary_files(list(files))
    draft = str(getattr(submission, "text", "") or "").strip()
    if not draft and not st.session_state.get(TEMP_FILES_KEY):
        st.toast("请输入问题或添加临时素材")
        return None
    return draft or "请阅读我已附加的临时素材，并结合当前对话继续分析。"


def _render_streaming_assistant_turn(
    service: TeachingAssistantService,
    *,
    prompt: str,
    profile: ProviderProfile,
    dataset_id: str | None,
    pure_mode: bool,
    temporary_materials: list[dict[str, Any]],
) -> TeachingTurnResult:
    tool_status = st.status(
        "正在分析问题并调用资料",
        expanded=True,
        type="compact",
    )
    steps_host = tool_status.container()
    content_slot = st.empty()
    content_parts: list[str] = []
    active_steps: dict[str, Any] = {}
    completed_tools = 0
    tools_collapsed = False
    result: TeachingTurnResult | None = None

    def collapse_tools() -> None:
        nonlocal tools_collapsed
        if tools_collapsed:
            return
        label = (
            f"已完成 {completed_tools} 项资料调用"
            if completed_tools
            else "分析完成"
        )
        tool_status.update(
            label=label,
            state="complete",
            expanded=False,
        )
        tools_collapsed = True

    try:
        for event in service.run_turn_stream(
            prompt=prompt,
            profile=profile,
            client=OpenAICompatibleAssistantClient(profile),
            dataset_id=dataset_id,
            pure_mode=pure_mode,
            temporary_materials=temporary_materials,
        ):
            if isinstance(event, TeachingToolStarted):
                tool_status.update(
                    label=f"正在调用：{event.label}",
                    state="running",
                    expanded=True,
                )
                with steps_host:
                    step = st.status(
                        event.label,
                        expanded=True,
                        state="running",
                        type="step",
                    )
                    active_steps[event.call_id] = step
                    if event.detail:
                        step.caption(event.detail)
            elif isinstance(event, TeachingToolFinished):
                step = active_steps.pop(event.call_id, None)
                if step is not None:
                    if event.success:
                        completed_tools += 1
                        step.update(
                            label=event.label,
                            state="complete",
                        )
                        if event.detail:
                            step.caption(event.detail)
                    else:
                        step.update(
                            label=f"{event.label}失败",
                            state="error",
                        )
            elif isinstance(event, TeachingContentDelta):
                collapse_tools()
                content_parts.append(event.text)
                content_slot.markdown("".join(content_parts))
            elif isinstance(event, TeachingTurnComplete):
                result = event.result
    except Exception:
        tool_status.update(
            label="生成失败",
            state="error",
            expanded=True,
        )
        raise

    collapse_tools()
    if result is None:
        raise AIRequestError("教学建议没有生成最终结果")
    if not content_parts:
        content_slot.markdown(result.message["content"])
    return result


def render_teaching_recommendations_page(
    service: TeachingAssistantService,
    learning: LearningDataService,
    profile: ProviderProfile | None,
) -> None:
    st.session_state.setdefault(HISTORY_LIMIT_KEY, HISTORY_PAGE_SIZE)
    st.session_state.setdefault(TEMP_FILES_KEY, [])
    with st.container(key="teaching_page_shell"):
        _render_question_dialog(service)
        copy_payload, live_turn_slot = _render_messages(service, profile)
        _render_pending_action(service, profile)
    retry_prompt = st.session_state.pop(RETRY_PROMPT_KEY, None)
    prompt = _render_composer(
        service,
        learning,
        profile,
        copy_payload,
    )
    if not prompt and isinstance(retry_prompt, str):
        prompt = retry_prompt
    if not prompt:
        return

    pure_mode = st.session_state.get(PURE_MODE_KEY) == "纯净模式"
    dataset_id = None
    if not pure_mode:
        dataset_id = st.session_state.get(DATASET_KEY)
    temporary_files = list(st.session_state.get(TEMP_FILES_KEY, []))
    result = None
    with live_turn_slot:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=":material/robot:"):
            try:
                assert profile is not None
                result = _render_streaming_assistant_turn(
                    service,
                    prompt=prompt,
                    profile=profile,
                    dataset_id=(
                        dataset_id if isinstance(dataset_id, str) else None
                    ),
                    pure_mode=pure_mode,
                    temporary_materials=temporary_files,
                )
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
