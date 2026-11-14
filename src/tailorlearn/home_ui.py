from __future__ import annotations

import streamlit as st

from tailorlearn.home_dashboard import HomeDashboardService
from tailorlearn.learning_data import LearningDataService

HOME_UI_VERSION = "2026-11-14-dashboard-v4"


NAVIGATION_CARDS: tuple[dict[str, str], ...] = (
    {
        "page": "学习内容梳理",
        "title": "学习内容梳理",
        "description": "查看学习内容、学习要求、材料出处与内容联系。",
        "icon": ":material/account_tree:",
    },
    {
        "page": "材料与数据",
        "title": "材料与数据",
        "description": "管理课本、课程标准、学生、测验节点与作答记录。",
        "icon": ":material/folder_managed:",
    },
    {
        "page": "教师工作台",
        "title": "教师工作台",
        "description": "集中查看待处理事项、复核任务与后续行动。",
        "icon": ":material/assignment:",
    },
    {
        "page": "学生分析",
        "title": "学生分析",
        "description": "查看学生表现、证据状态、趋势条件与可追溯记录。",
        "icon": ":material/person_search:",
    },
    {
        "page": "班级分析",
        "title": "班级分析",
        "description": "比较班级表现并识别共同观察点与差异。",
        "icon": ":material/groups:",
    },
    {
        "page": "教学智能体",
        "title": "教学智能体",
        "description": "在配置 AI 接入后，围绕已有材料与数据进行教学对话。",
        "icon": ":material/smart_toy:",
    },
)


def _route_to_todo(todo: dict[str, object]) -> None:
    st.session_state["workbench_pending_route"] = {
        "target_page": todo["target_page"],
        "context": todo["context"],
    }
    st.session_state["active_page"] = todo["target_page"]
    st.rerun()


def render_navigation_home(
    learning: LearningDataService,
    dashboard: HomeDashboardService,
) -> None:
    st.subheader("今日教学总览")
    st.caption("所有观察均以选定数据集中的有效证据为准；信息不足仅提示补充观察。")
    datasets = learning.list_datasets()
    if not datasets:
        st.info("尚无学情数据集。请先在“材料与数据”导入数据。")
        return
    dataset_ids = [str(item["id"]) for item in datasets]
    current = st.session_state.get("home_dataset")
    if current not in dataset_ids:
        current = dataset_ids[0]
        st.session_state["home_dataset"] = current
    dataset_id = st.selectbox(
        "学情数据集",
        dataset_ids,
        key="home_dataset",
        format_func=lambda value: next(
            (item["name"] for item in datasets if item["id"] == value), value
        ),
    )
    summary = dashboard.build(dataset_id)
    metrics = summary["metrics"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("学生数", metrics["student_count"], border=True)
    metric_columns[1].metric(
        "有效已评分作答", metrics["valid_scored_answers"], border=True
    )
    metric_columns[2].metric(
        "数据处理待办", metrics["data_processing_todos"], border=True
    )
    metric_columns[3].metric(
        "补充观察/复核", metrics["follow_up_todos"], border=True
    )

    overview, evidence = st.columns((3, 2), gap="medium")
    with overview:
        st.subheader("优先待办")
        todos = list(summary["todos"])[:5]
        if not todos:
            st.success("当前没有需要处理的聚合待办。")
        for index, todo in enumerate(todos):
            with st.container(border=True):
                st.markdown(f"**{todo['title']}**")
                st.caption(str(todo["reason"]))
                if st.button(
                    "去处理",
                    icon=":material/arrow_forward:",
                    key=f"home_todo_{dataset_id}_{index}",
                    type="tertiary",
                ):
                    _route_to_todo(todo)
    with evidence:
        st.subheader("证据覆盖概览")
        coverage = summary["evidence_overview"]
        sufficient_column, observation_column = st.columns(2)
        sufficient_column.metric(
            "证据充分主题",
            coverage["sufficient"],
            help="在当前数据集范围内具备充分有效、已评分证据的主题数。",
        )
        observation_column.metric(
            "待补充观察主题",
            coverage["needs_observation"],
            help="当前证据不足、待评分、未作答或异常数据只提示补充观察。",
        )
        with st.container(border=True):
            for item in coverage["themes"]:
                status = (
                    "证据充分"
                    if item["evidence_status"] == "sufficient"
                    else "待补充观察"
                )
                st.markdown(f"{item['theme_name']}　:{'blue' if status == '证据充分' else 'gray'}-badge[{status}]")

        st.subheader("班级证据覆盖")
        class_coverage = coverage.get("class_coverage") or []
        if not class_coverage:
            st.caption("当前数据集暂无班级维度的证据记录。")
        else:
            with st.container(border=True):
                for item in class_coverage:
                    st.markdown(
                        f"{item['class_code']}　"
                        f":blue-badge[充分 {item['sufficient']}] "
                        f":gray-badge[待观察 {item['needs_observation']}] "
                        f"<span style='opacity:.65'>· {item['scored_records']} 条有效评分</span>",
                        unsafe_allow_html=True,
                    )

        st.subheader("待办分布")
        todos_by_page = summary.get("todos_by_page") or {}
        if not todos_by_page:
            st.caption("当前没有需要处理的聚合待办。")
        else:
            with st.container(border=True):
                for page, count in sorted(
                    todos_by_page.items(), key=lambda pair: (-pair[1], pair[0])
                ):
                    st.markdown(
                        f"{page}　:blue-badge[{count} 项]",
                    )
        st.caption("待补充观察表示当前证据不足，不构成学习掌握结论。")

    st.subheader("常用功能")

    for start in range(0, len(NAVIGATION_CARDS), 2):
        columns = st.columns(2, gap="medium")
        for column, card in zip(columns, NAVIGATION_CARDS[start : start + 2]):
            with column:
                with st.container(border=True):
                    st.markdown(f"#### {card['title']}")
                    st.caption(card["description"])
                    if st.button(
                        "进入",
                        icon=card["icon"],
                        key=f"home_open_{card['page']}",
                        width="stretch",
                    ):
                        st.session_state["active_page"] = card["page"]
                        st.rerun()
