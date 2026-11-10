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
    "学情数据": PageMeta(
        label="学情数据",
        icon="database.svg",
        eyebrow="数据准备",
        description="集中管理六类学情 CSV，完成字段映射、重复合并、质量检查与变更追溯。",
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
    "学科分析": PageMeta(
        label="学科分析",
        icon="books.svg",
        eyebrow="材料准备",
        description="导入课标与教材，完成文字提取、章节切分、AI 结构化和逐单元人工审核。",
    ),
    "AI 参数设置": PageMeta(
        label="AI 参数设置",
        icon="settings.svg",
        eyebrow="系统配置",
        description="管理本机 AI 厂商配置。日常工作以内容证据为主，参数仅在使用 AI 能力时需要维护。",
    ),
}

NAV_ORDER = list(PAGE_META)


def apply_app_style() -> None:
    st.html(
        """
        <style>
        .tl-brand {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            padding: 0.2rem 0.15rem 0.9rem;
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
        .st-key-main_navigation [role="radiogroup"] {
            gap: 0.28rem;
        }
        .st-key-main_navigation [role="radiogroup"] label {
            width: 100%;
            padding: 0.52rem 0.62rem;
            border: 1px solid transparent;
            border-radius: 0.65rem;
            transition: background-color 120ms ease, border-color 120ms ease;
        }
        .st-key-main_navigation [role="radiogroup"] label:hover {
            background: #EEF1F6;
            border-color: #E1E5EB;
        }
        .st-key-main_navigation [role="radiogroup"] label:has(input:checked) {
            background: #EDF2FF;
            border-color: #D6E1FF;
        }
        .st-key-main_navigation img {
            width: 1.12rem;
            height: 1.12rem;
            object-fit: contain;
            margin-right: 0.18rem;
            vertical-align: -0.2rem;
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


def navigation_label(page: str) -> str:
    meta = PAGE_META[page]
    return f"![]({'app/static/icons/' + meta.icon}) {page}"


def render_sidebar_navigation() -> str:
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
        st.caption("工作流")
        page = st.radio(
            "导航",
            NAV_ORDER,
            format_func=navigation_label,
            label_visibility="collapsed",
            key="main_navigation",
            width="stretch",
        )
        st.space("small")
        st.caption("证据优先 · AI 草稿 · 教师复核")
    return str(page)


def render_page_header(page: str) -> None:
    meta = PAGE_META[page]
    with st.container(key="page_header"):
        st.markdown(f":blue-badge[{meta.eyebrow}]")
        st.title(meta.label)
        st.caption(meta.description)
