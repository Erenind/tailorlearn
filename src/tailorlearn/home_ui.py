from __future__ import annotations

import streamlit as st


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
        "page": "教学建议",
        "title": "教学建议",
        "description": "基于证据形成分层活动、材料建议和教师复核。",
        "icon": ":material/lightbulb:",
    },
    {
        "page": "教学智能体",
        "title": "教学智能体",
        "description": "在配置 AI 接入后，围绕已有材料与数据进行教学对话。",
        "icon": ":material/smart_toy:",
    },
)


def render_navigation_home() -> None:
    st.subheader("从这里开始")
    st.caption("选择一个入口开始工作。学习结论始终以已有证据和教师复核为准。")

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
