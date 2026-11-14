from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

UI_VERSION = "2026-11-14-navigation-without-teaching-page"


@dataclass(frozen=True)
class PageMeta:
    label: str
    icon: str


PAGE_META: dict[str, PageMeta] = {
    "导航首页": PageMeta(
        label="导航首页",
        icon="logo.svg",
    ),
    "学习内容梳理": PageMeta(
        label="学习内容梳理",
        icon="outline.svg",
    ),
    "材料与数据": PageMeta(
        label="材料与数据",
        icon="material_center.svg",
    ),
    "学生分析": PageMeta(
        label="学生分析",
        icon="student.svg",
    ),
    "班级分析": PageMeta(
        label="班级分析",
        icon="class.svg",
    ),
    "教学智能体": PageMeta(
        label="教学智能体",
        icon="teaching.svg",
    ),
    "教师工作台": PageMeta(
        label="教师工作台",
        icon="database.svg",
    ),
    "AI 参数设置": PageMeta(
        label="AI 参数设置",
        icon="settings.svg",
    ),
}

NAV_GROUPS: dict[str, tuple[str, ...]] = {
    "开始使用": ("导航首页",),
    "学习准备": ("学习内容梳理",),
    "材料与管理": ("材料与数据", "教师工作台"),
    "学情分析": ("学生分析", "班级分析"),
    "教学支持": ("教学智能体", "AI 参数设置"),
}
NAV_ORDER = [page for pages in NAV_GROUPS.values() for page in pages]
PAGE_SLUGS = {
    "导航首页": "home",
    "学习内容梳理": "outline",
    "材料与数据": "material_center",
    "学生分析": "student",
    "班级分析": "class",
    "教学智能体": "teaching_agent",
    "教师工作台": "workbench",
    "AI 参数设置": "ai_settings",
}


def apply_app_style() -> None:
    st.html(
        """
        <style>
        .tl-brand {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            padding: 0 0.15rem 0.78rem;
        }
        .tl-brand img {
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 0.7rem;
        }
        .tl-brand-title {
            color: #172033;
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.2;
            letter-spacing: -0.01em;
        }
        .tl-brand-subtitle {
            color: #7A8496;
            font-size: 0.72rem;
            line-height: 1.35;
            margin-top: 0.16rem;
        }
        .st-key-app_nav {
            gap: 0.22rem;
        }
        .st-key-app_nav button {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            min-height: 2.35rem;
            padding: 0.48rem 0.62rem;
            border: 1px solid transparent;
            border-radius: 0.65rem;
            background: transparent;
            color: #4B5565;
            font-size: 14px;
            font-weight: 450;
            box-shadow: none;
            text-align: left;
            transition:
                background-color 120ms ease,
                border-color 120ms ease,
                color 120ms ease;
        }
        .st-key-app_nav button > div {
            width: 100%;
            justify-content: flex-start;
        }
        .st-key-app_nav button:hover {
            background: #EEF1F6;
            border-color: #E1E5EB;
            color: #25324A;
        }
        .st-key-app_nav button p {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 8px;
            width: 100%;
            margin: 0;
            font-size: 14px;
            line-height: 1.35;
            text-align: left;
        }
        .st-key-app_nav img {
            display: block;
            flex: 0 0 16px;
            width: 16px;
            height: 16px;
            margin: 0;
            object-fit: contain;
        }
        .st-key-material_center_nav {
            gap: 0.35rem;
            padding: 0.75rem;
            border: 1px solid #E1E5EB;
            border-radius: 1rem;
            background: #FFFFFF;
        }
        .st-key-material_center_nav button {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            min-height: 2.35rem;
            padding: 0.45rem 0.68rem;
            border: 1px solid transparent;
            border-radius: 0.65rem;
            background: transparent;
            color: #4B5565;
            font-size: 14px;
            font-weight: 450;
            box-shadow: none;
            transition:
                background-color 120ms ease,
                border-color 120ms ease,
                color 120ms ease;
        }
        .st-key-material_center_nav button:hover {
            background: #EEF1F6;
            border-color: #E1E5EB;
            color: #25324A;
        }
        .st-key-material_center_nav button p {
            display: flex;
            align-items: center;
            gap: 7px;
            margin: 0;
            font-size: 14px;
            line-height: 1.35;
            text-align: left;
        }
        .st-key-material_center_nav img {
            display: block;
            flex: 0 0 16px;
            width: 16px;
            height: 16px;
            margin: 0;
            object-fit: contain;
        }
        .st-key-material_browser_files [class*="st-key-material_catalog_card_"] {
            min-height: 8.5rem;
            padding: 0.9rem;
            border: 1px solid #E1E5EB;
            border-radius: 0.9rem;
            background: #FFFFFF;
        }
        .st-key-material_directory {
            position: sticky;
            top: 1rem;
            padding: 0.9rem;
            border: 1px solid #E1E5EB;
            border-radius: 0.9rem;
            background: #F7F8FB;
        }
        .st-key-material_unit_blocks [class*="st-key-material_unit_block_"] {
            padding: 1rem 1.1rem;
            border: 1px solid #E1E5EB;
            border-radius: 0.9rem;
            background: #F7F8FB;
        }
        [class*="st-key-material_unit_title_"] {
            align-self: center;
        }
        [class*="st-key-material_unit_title_"] p {
            margin: 0;
            line-height: 1.5rem;
        }
        [class*="st-key-material_manage_content_select_"] {
            align-self: center;
            min-height: 1.5rem;
        }
        [class*="st-key-material_manage_content_select_"] label {
            display: flex;
            align-items: center;
            min-height: 1.5rem;
            margin: 0;
            padding: 0;
        }
        .st-key-provider_summary {
            margin-bottom: 0.35rem;
        }
        </style>
        """
    )


def navigation_key(page: str) -> str:
    return f"nav_{PAGE_SLUGS[page]}"


def default_page() -> str:
    return NAV_ORDER[0]


def navigation_label(page: str, *, active: bool) -> str:
    meta = PAGE_META[page]
    title = f"**{page}**" if active else page
    return f"![]({'app/static/icons/' + meta.icon}) {title}"


def render_sidebar_navigation() -> str:
    active_page = st.session_state.get("active_page")
    if active_page not in NAV_ORDER:
        legacy_page = st.session_state.get("main_navigation")
        active_page = legacy_page if legacy_page in NAV_ORDER else default_page()
        st.session_state["active_page"] = active_page

    with st.sidebar:
        st.html(
            """
            <div class="tl-brand">
                <img src="app/static/icons/logo.svg" alt="">
                <div>
                    <div class="tl-brand-title">TailorLearn</div>
                    <div class="tl-brand-subtitle">教师教学分析工作台</div>
                </div>
            </div>
            """
        )
        with st.container(key="app_nav"):
            for group_label, pages in NAV_GROUPS.items():
                st.caption(group_label)
                for page in pages:
                    if st.button(
                        navigation_label(page, active=page == active_page),
                        key=navigation_key(page),
                        type="secondary",
                        width="stretch",
                    ):
                        st.session_state["active_page"] = page
                        st.rerun()
        active_key = navigation_key(str(active_page))
        st.html(
            f"""
            <style>
            .st-key-{active_key} button {{
                background: #EDF2FF !important;
                border-color: #D6E1FF !important;
                color: #1F46C5 !important;
                font-weight: 700;
                box-shadow: inset 3px 0 0 #2F5BEA;
            }}
            .st-key-{active_key} button p {{
                font-weight: 700;
            }}
            </style>
            """
        )
    return str(active_page)


def render_page_header(page: str) -> None:
    meta = PAGE_META[page]
    st.title(meta.label)
