from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st

from tailorlearn.learning_data import LearningDataService
from tailorlearn.teacher_workbench import TeacherWorkbenchService


CATEGORY_META = {
    "data_processing": ("数据处理", "先完成评分或核验异常记录"),
    "supplemental_observation": ("补充观察", "信息不足时只安排补充观察，不作掌握判定"),
    "ai_review": ("AI 待复核", "AI 草稿须经教师复核后使用"),
    "needs_update": ("需要更新", "数据变更后，原草稿需重新核验或生成"),
}


def _dataset_selector(learning: LearningDataService) -> dict[str, Any] | None:
    datasets = learning.list_datasets()
    if not datasets:
        st.info("尚无学情数据集。请先在“材料与数据”导入数据。")
        return None
    dataset_id = st.selectbox(
        "学情数据集",
        [item["id"] for item in datasets],
        key="workbench_dataset",
        format_func=lambda value: next(
            (item["name"] for item in datasets if item["id"] == value), value
        ),
    )
    return next(item for item in datasets if item["id"] == dataset_id)


def _route_to_todo(todo: dict[str, Any]) -> None:
    st.session_state["workbench_pending_route"] = {
        "target_page": todo["target_page"],
        "context": todo["context"],
    }
    st.session_state["active_page"] = todo["target_page"]
    st.rerun()


def _todo_frame(todos: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "待办": item["title"],
                "说明": item["reason"],
                "前往": item["target_page"],
            }
            for item in todos
        ]
    )


def render_teacher_workbench_page(
    service: TeacherWorkbenchService,
    learning: LearningDataService,
) -> None:
    st.caption(
        "工作台只聚合已有待办并跳转至原页面处理；"
        "待评分、未作答、异常与证据不足均不等同于未掌握。"
    )
    dataset = _dataset_selector(learning)
    if dataset is None:
        return
    board = service.build_todos(dataset["id"])
    summary = board["summary"]

    st.subheader("待办概览")
    with st.grid("auto", min_column_width=165, gap="small"):
        for category in CATEGORY_META:
            label, help_text = CATEGORY_META[category]
            st.metric(label, summary[category], border=True, help=help_text)

    if not board["todos"]:
        st.success("当前没有需要处理的聚合待办。")
        return

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for todo in board["todos"]:
        grouped[str(todo["category"])].append(todo)

    for category in CATEGORY_META:
        todos = grouped[category]
        if not todos:
            continue
        label, help_text = CATEGORY_META[category]
        st.subheader(f"{label}（{len(todos)}）")
        st.caption(help_text)
        st.dataframe(
            _todo_frame(todos),
            hide_index=True,
            width="stretch",
            key=f"workbench_table_{category}_{dataset['id']}",
        )
        columns = st.columns(min(3, len(todos)))
        for index, todo in enumerate(todos):
            with columns[index % len(columns)]:
                if st.button(
                    f"处理：{todo['title']}",
                    key=f"workbench_route_{category}_{index}_{dataset['id']}",
                    width="stretch",
                ):
                    _route_to_todo(todo)
