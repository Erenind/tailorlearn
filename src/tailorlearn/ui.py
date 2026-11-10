from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class PageMeta:
    label: str
    icon: str
    eyebrow: str
    description: str


PAGE_META: dict[str, PageMeta] = {
    "学习内容梳理": PageMeta(
        label="学习内容梳理",
        icon="outline.svg",
        eyebrow="任务 1 · 内容基准",
        description="按主题梳理学习内容、课标要求、必要联系与教材出处，并生成可核对的成果包。",
    ),
    "材料中心": PageMeta(
        label="材料中心",
        icon="material_center.svg",
        eyebrow="材料与学情",
        description="集中管理课本、课程标准与六类学情数据，支持浏览、编辑、导入和作答异常检测。",
    ),
    "学生分析": PageMeta(
        label="学生分析",
        icon="student.svg",
        eyebrow="任务 2 · 个体表现",
        description="从有效作答中观察学生表现、证据充分度与节点变化，信息不足时不作负面判定。",
    ),
    "班级分析": PageMeta(
        label="班级分析",
        icon="class.svg",
        eyebrow="任务 2 · 群体表现",
        description="比较班级主题表现与证据覆盖，识别下一步最值得补充观察的共同主题。",
    ),
    "教学建议": PageMeta(
        label="教学建议",
        icon="teaching.svg",
        eyebrow="任务 3 · 教学决策",
        description="基于班级证据、分层观察和现有材料生成可审核的教学建议与脱敏案例。",
    ),
    "AI 参数设置": PageMeta(
        label="AI 参数设置",
        icon="settings.svg",
        eyebrow="系统配置",
        description="管理本机 AI 厂商配置。日常工作以内容证据为主，参数仅在使用 AI 能力时需要维护。",
    ),
}

NAV_ORDER = list(PAGE_META)
PAGE_SLUGS = {
    "学习内容梳理": "outline",
    "材料中心": "material_center",
    "学生分析": "student",
    "班级分析": "class",
    "教学建议": "teaching",
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
        .st-key-page_header {
            padding: 1.08rem 1.22rem 0.92rem;
            border: 1px solid #E1E5EB;
            border-radius: 1rem;
            background: #FFFFFF;
            box-shadow: 0 1px 2px rgba(24, 34, 48, 0.04);
            margin-bottom: 0.8rem;
        }
        .st-key-page_header h1 {
            margin: 0.28rem 0 0.2rem;
            color: #172033;
            font-size: 1.78rem;
            line-height: 1.2;
            letter-spacing: -0.025em;
        }
        .st-key-page_header p {
            max-width: 72rem;
            margin-bottom: 0;
            color: #667085;
            font-size: 0.9rem;
            line-height: 1.55;
        }
        .st-key-provider_summary {
            margin-bottom: 0.35rem;
        }
        </style>
        """
    )


def navigation_key(page: str) -> str:
    return f"nav_{PAGE_SLUGS[page]}"


def navigation_label(page: str, *, active: bool) -> str:
    meta = PAGE_META[page]
    title = f"**{page}**" if active else page
    return f"![]({'app/static/icons/' + meta.icon}) {title}"


def render_sidebar_navigation() -> str:
    active_page = st.session_state.get("active_page")
    if active_page not in NAV_ORDER:
        legacy_page = st.session_state.get("main_navigation")
        active_page = legacy_page if legacy_page in NAV_ORDER else NAV_ORDER[0]
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
            for page in NAV_ORDER:
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
    with st.container(key="page_header"):
        st.markdown(f":blue-badge[{meta.eyebrow}]")
        st.title(meta.label)
        st.caption(meta.description)
