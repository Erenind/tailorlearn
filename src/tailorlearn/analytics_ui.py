from __future__ import annotations

import csv
from io import StringIO
import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tailorlearn.analytics import AnalyticsService
from tailorlearn.ai import AIRequestError, OpenAICompatibleClient
from tailorlearn.learning_data import LearningDataService
from tailorlearn.models import ProviderProfile
from tailorlearn.student_diagnosis import StudentDiagnosisService


ANALYTICS_UI_VERSION = "2026-11-13-student-ranking-primary-action"
STUDENT_ANALYSIS_SECTIONS = ("数据概览", "问题分析")
CLASS_ANALYSIS_SECTIONS = ("班级对比", "具体班级分析")
STUDENT_RANKING_SORT_OPTIONS = (
    "综合得分降序",
    "综合得分升序",
    "按学生编号",
)
STUDENT_DATASET_STATE_KEY = "analytics_student_dataset_selected"
STUDENT_DETAIL_STATE_KEY = "analytics_student_detail_code"
EVIDENCE_LABELS = {
    "sufficient": "证据充分",
    "limited": "证据有限",
    "insufficient": "信息不足",
    "unavailable": "不可用",
}


def _datasets(learning: LearningDataService) -> list[dict[str, Any]]:
    return learning.list_datasets()


def _dataset_selector(
    learning: LearningDataService,
    *,
    key: str,
) -> dict[str, Any] | None:
    datasets = _datasets(learning)
    if not datasets:
        st.info("尚无学情数据集。请先在“材料与数据”导入数据。")
        return None
    dataset_id = st.selectbox(
        "学情数据集",
        [row["id"] for row in datasets],
        key=key,
        format_func=lambda value: next(
            (row["name"] for row in datasets if row["id"] == value), value
        ),
    )
    return next(row for row in datasets if row["id"] == dataset_id)


def _node_selector(
    learning: LearningDataService,
    dataset_id: str,
    *,
    key: str,
    include_all: bool = True,
) -> str | None:
    nodes = learning.list_records(dataset_id, "node")
    options = (["__all__"] if include_all else []) + [
        str(row["node_code"]) for row in nodes
    ]
    if not options:
        return None
    selected = st.selectbox(
        "测验节点",
        options,
        key=key,
        format_func=lambda value: (
            "全部节点" if value == "__all__" else next(
                (
                    f"{row['node_code']} · {row.get('test_date') or '未标日期'}"
                    for row in nodes
                    if row["node_code"] == value
                ),
                value,
            )
        ),
    )
    return None if selected == "__all__" else selected


def _metric_table(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "主题/指标": metric.get("theme_name") or metric.get("indicator_name"),
            "得分率": (
                f"{metric['score_rate']:.1%}"
                if metric.get("score_rate") is not None
                else "—"
            ),
            "有效记录": metric.get("scored_records", 0),
            "待评分": metric.get("pending_records", 0),
            "未作答": metric.get("unanswered_records", 0),
            "题目数": metric.get("material_count", "—"),
            "证据状态": EVIDENCE_LABELS.get(
                str(metric.get("evidence_status")),
                str(metric.get("evidence_status") or ""),
            ),
        }
        for metric in metrics
    ]


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + output.getvalue()


def _json_text(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _render_ai_diagnosis_tab(
    diagnosis: StudentDiagnosisService,
    profile: ProviderProfile | None,
    *,
    dataset_id: str,
    student_code: str,
    class_code: str,
    node_code: str | None,
) -> None:
    st.subheader("AI 诊断与教师复核")
    st.caption("AI 仅解释已核验的证据包；信息不足时只提出补充观察，不判定未掌握。")
    if node_code is None:
        st.info("请选择一个具体测验节点，再生成或查看 AI 诊断。")
        return
    diagnosis.refresh_stale_diagnoses(dataset_id)
    latest = diagnosis.get_latest_diagnosis(dataset_id, student_code, node_code)
    if profile is None:
        st.info("请先在“AI 参数设置”保存并启用厂商配置；历史草稿仍可查看。")
    else:
        single, batch = st.columns(2)
        if single.button("为当前学生生成草稿", key="diagnosis_generate_student"):
            try:
                with st.spinner("正在生成可复核的学生诊断草稿..."):
                    job = diagnosis.create_student_job(
                        dataset_id, student_code, node_code, profile
                    )
                    result = diagnosis.process_job(
                        job["id"], OpenAICompatibleClient(profile)
                    )
                if result["status"] == "completed":
                    st.success("学生诊断草稿已生成，等待教师复核。")
                else:
                    st.warning("任务已保存，但存在失败项；可稍后继续处理。")
            except (AIRequestError, ValueError) as exc:
                st.error(f"无法生成诊断草稿：{exc}")
        if batch.button(
            f"为 {class_code} 批量生成", key="diagnosis_generate_class"
        ):
            try:
                with st.spinner("正在逐名生成班级学生诊断，可在中断后继续处理..."):
                    job = diagnosis.create_batch_job(
                        dataset_id, class_code, node_code, profile
                    )
                    result = diagnosis.process_job(
                        job["id"], OpenAICompatibleClient(profile)
                    )
                if result["status"] == "completed":
                    st.success(f"已完成 {result['completed']} 名学生的诊断草稿。")
                else:
                    st.warning(
                        f"已完成 {result['completed']} 名，失败 {result['failed']} 名；"
                        "任务状态已保留。"
                    )
            except (AIRequestError, ValueError) as exc:
                st.error(f"无法创建批量诊断任务：{exc}")
        jobs = diagnosis.list_jobs(dataset_id, class_code, node_code)
        if jobs:
            latest_job = jobs[0]
            st.caption(
                "当前班级任务："
                f"已完成 {latest_job['completed']}/{latest_job['total']}，"
                f"失败 {latest_job['failed']}。"
            )
            st.progress(
                latest_job["completed"] / latest_job["total"]
                if latest_job["total"]
                else 0
            )
            if latest_job["status"] in {"queued", "running", "completed_with_errors"}:
                if st.button("继续处理当前班级任务", key="diagnosis_resume_class"):
                    try:
                        if latest_job["status"] == "completed_with_errors":
                            diagnosis.retry_failed_items(latest_job["id"])
                        with st.spinner("正在继续处理已保存的诊断任务..."):
                            result = diagnosis.process_job(
                                latest_job["id"], OpenAICompatibleClient(profile)
                            )
                        st.success(
                            f"任务处理完成：成功 {result['completed']} 名，"
                            f"失败 {result['failed']} 名。"
                        )
                    except (AIRequestError, ValueError) as exc:
                        st.error(f"无法继续任务：{exc}")
    if latest is None:
        st.info("尚无当前节点的 AI 诊断草稿。")
        return
    status_labels = {
        "needs_review": "待教师复核",
        "accepted": "已采纳",
        "rejected": "已拒绝",
        "stale": "数据已更新，需重新生成",
    }
    st.info("当前草稿状态：" + status_labels.get(latest["status"], latest["status"]))
    draft = latest["draft"]
    st.markdown("**AI 谨慎摘要**")
    st.write(draft["summary"])
    if draft["observations"]:
        st.markdown("**已观察到的表现与依据**")
        st.dataframe(draft["observations"], hide_index=True, width="stretch")
    if draft["information_gaps"]:
        st.markdown("**信息不足与补充观察**")
        st.dataframe(draft["information_gaps"], hide_index=True, width="stretch")
    if draft["suggestions"]:
        st.markdown("**仅限已有材料的建议**")
        st.dataframe(draft["suggestions"], hide_index=True, width="stretch")
    if draft["teacher_actions"]:
        st.markdown("**建议教师操作**")
        st.write("\n".join(f"- {item}" for item in draft["teacher_actions"]))
    reviewed_summary = st.text_area(
        "教师修订后的摘要",
        value=latest["reviewed_summary"] or draft["summary"],
        key=f"diagnosis_reviewed_summary_{latest['id']}",
    )
    review_note = st.text_area(
        "审核备注",
        value=latest["review_note"],
        key=f"diagnosis_review_note_{latest['id']}",
    )
    accept, reject = st.columns(2)
    if accept.button("采纳教师修订", key=f"diagnosis_accept_{latest['id']}"):
        diagnosis.review_diagnosis(
            latest["id"],
            status="accepted",
            reviewed_summary=reviewed_summary,
            review_note=review_note,
        )
        st.success("已采纳并保留教师修订。")
    if reject.button("拒绝此草稿", key=f"diagnosis_reject_{latest['id']}"):
        diagnosis.review_diagnosis(
            latest["id"],
            status="rejected",
            reviewed_summary=reviewed_summary,
            review_note=review_note,
        )
        st.info("草稿已拒绝，原始响应仍保留以便追溯。")


def _render_analysis_top_nav(
    options: tuple[tuple[str, str, str], ...],
    *,
    state_key: str,
    container_key: str,
    button_key_prefix: str,
) -> str:
    sections = tuple(section for section, _, _ in options)
    suffix_by_section = {
        section: suffix for section, suffix, _ in options
    }
    active = st.session_state.get(state_key)
    if active not in sections:
        active = sections[0]
        st.session_state[state_key] = active
    with st.container(
        horizontal=True,
        wrap=False,
        gap="xsmall",
        key=container_key,
    ):
        for section, suffix, icon in options:
            if st.button(
                f"{icon} {section}",
                key=f"{button_key_prefix}_{suffix}",
                type="secondary",
                width="content",
            ):
                st.session_state[state_key] = section
                st.rerun()
    active_suffix = suffix_by_section[str(active)]
    st.html(
        f"""
        <style>
        .st-key-{container_key} {{
            padding: 0.35rem;
            border: 1px solid #E1E5EB;
            border-radius: 0.9rem;
            background: #FFFFFF;
        }}
        .st-key-{button_key_prefix}_{active_suffix} button {{
            background: #EDF2FF !important;
            border-color: #D6E1FF !important;
            color: #1F46C5 !important;
            font-weight: 700;
            box-shadow: inset 0 -2px 0 #2F5BEA;
        }}
        .st-key-{button_key_prefix}_{active_suffix} button p {{
            font-weight: 700;
        }}
        </style>
        """
    )
    return str(active)


def _render_student_analysis_top_nav() -> str:
    return _render_analysis_top_nav(
        (
            ("数据概览", "overview", ":material/dashboard:"),
            ("问题分析", "issues", ":material/troubleshoot:"),
        ),
        state_key="student_analysis_section",
        container_key="student_analysis_top_nav",
        button_key_prefix="student_analysis_nav",
    )


def _render_class_analysis_top_nav() -> str:
    return _render_analysis_top_nav(
        (
            ("班级对比", "compare", ":material/compare_arrows:"),
            ("具体班级分析", "specific", ":material/groups:"),
        ),
        state_key="class_analysis_section",
        container_key="class_analysis_top_nav",
        button_key_prefix="class_analysis_nav",
    )


def _render_student_dataset_context(
    learning: LearningDataService,
) -> dict[str, Any] | None:
    datasets = _datasets(learning)
    dataset_ids = {str(row["id"]) for row in datasets}
    selected_id = st.session_state.get(STUDENT_DATASET_STATE_KEY)
    if selected_id not in dataset_ids:
        selected_id = None
        st.session_state.pop(STUDENT_DATASET_STATE_KEY, None)
        st.session_state.pop(STUDENT_DETAIL_STATE_KEY, None)

    if selected_id is None:
        st.markdown("**学情数据集**")
        st.caption("选择要分析的数据集，进入后查看学生排名榜单。")
        if not datasets:
            st.info("尚无学情数据集。请先在“材料与数据”导入数据。")
            return None
        with st.container(key="analytics_student_dataset_catalog"):
            with st.grid("auto", min_column_width=260, gap="small"):
                for dataset in datasets:
                    student_count = len(
                        learning.list_records(dataset["id"], "student")
                    )
                    answer_count = len(
                        learning.list_records(dataset["id"], "answer")
                    )
                    with st.container(
                        border=True,
                        key=f"analytics_student_dataset_card_{dataset['id']}",
                    ):
                        st.markdown(f"**{dataset['name']}**")
                        if dataset.get("description"):
                            st.caption(dataset["description"])
                        st.caption(
                            f"学生 {student_count} · 作答 {answer_count}"
                        )
                        if st.button(
                            "进入分析",
                            icon=":material/arrow_forward:",
                            key=(
                                "analytics_student_dataset_open_"
                                f"{dataset['id']}"
                            ),
                            width="stretch",
                        ):
                            st.session_state[STUDENT_DATASET_STATE_KEY] = dataset[
                                "id"
                            ]
                            st.session_state.pop(STUDENT_DETAIL_STATE_KEY, None)
                            st.rerun()
        return None

    dataset = next(
        row for row in datasets if str(row["id"]) == str(selected_id)
    )
    detail_code = st.session_state.get(STUDENT_DETAIL_STATE_KEY)
    with st.container(
        horizontal=True,
        vertical_alignment="center",
        key="analytics_student_dataset_path",
    ):
        if st.button(
            "学情数据集",
            type="tertiary",
            key="analytics_student_path_datasets",
        ):
            st.session_state.pop(STUDENT_DATASET_STATE_KEY, None)
            st.session_state.pop(STUDENT_DETAIL_STATE_KEY, None)
            st.rerun()
        st.markdown("/")
        st.markdown(f"**{dataset['name']}**")
        if detail_code:
            if st.button(
                "返回榜单",
                icon=":material/arrow_back:",
                type="tertiary",
                key=f"analytics_student_back_to_rankings_{dataset['id']}",
            ):
                st.session_state.pop(STUDENT_DETAIL_STATE_KEY, None)
                st.rerun()
            st.markdown("/")
            st.markdown(f"**{detail_code}**")
    return dataset


def _ranking_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "综合名次": (
                    str(row["rank"])
                    if row["rank"] is not None
                    else "暂无排名"
                ),
                "学生编号": row["student_code"],
                "班级": row["class_code"],
                "综合得分率": row["score_rate"],
                "已观察主题": (
                    f"{row['observed_theme_count']}/{row['total_theme_count']}"
                ),
                "记录完整度": (
                    f"{row['complete_theme_count']}/"
                    f"{row['total_theme_count']}"
                    f"（{row['record_completeness']:.0%}）"
                ),
                "证据状态": EVIDENCE_LABELS[row["evidence_status"]],
                "查看": ":material/arrow_forward: 查看",
            }
            for row in rows
        ]
    )


def _incomplete_ranking_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "学生编号": row["student_code"],
                "班级": row["class_code"],
                "记录完整度": (
                    f"{row['complete_theme_count']}/"
                    f"{row['total_theme_count']}"
                    f"（{row['record_completeness']:.0%}）"
                ),
                "已观察主题": (
                    f"{row['observed_theme_count']}/{row['total_theme_count']}"
                ),
                "已观察主题得分率": row["score_rate"],
                "证据状态": EVIDENCE_LABELS[row["evidence_status"]],
                "查看": ":material/arrow_forward: 查看",
            }
            for row in rows
        ]
    )


def _sorted_ranking(
    rows: list[dict[str, Any]],
    sort_mode: str,
) -> list[dict[str, Any]]:
    scored = [row for row in rows if row["score_rate"] is not None]
    unscored = sorted(
        (row for row in rows if row["score_rate"] is None),
        key=lambda row: str(row["student_code"]),
    )
    if sort_mode == "综合得分升序":
        scored = sorted(
            scored,
            key=lambda row: (
                float(row["score_rate"]),
                str(row["student_code"]),
            ),
        )
    elif sort_mode == "按学生编号":
        scored = sorted(scored, key=lambda row: str(row["student_code"]))
    else:
        scored = sorted(
            scored,
            key=lambda row: (
                -float(row["score_rate"]),
                str(row["student_code"]),
            ),
        )
    return [*scored, *unscored]


def _student_code_from_click(
    click: Any,
    student_codes: list[str],
) -> str | None:
    if click is None:
        return None
    try:
        row_position = int(click["row"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= row_position < len(student_codes):
        return None
    return str(student_codes[row_position])


def _open_student_detail(
    *,
    click_key: str,
    dataset_id: str,
    student_codes: list[str],
) -> None:
    student_code = _student_code_from_click(
        st.session_state.get(click_key),
        student_codes,
    )
    if student_code is None:
        return
    st.session_state[STUDENT_DATASET_STATE_KEY] = dataset_id
    st.session_state[STUDENT_DETAIL_STATE_KEY] = student_code


def _render_student_rankings(
    dataset_id: str,
    ranking: list[dict[str, Any]],
) -> None:
    sort_key = f"analytics_student_ranking_sort_{dataset_id}"
    with st.container(
        border=True,
        key=f"analytics_student_ranking_card_{dataset_id}",
    ):
        heading, sort_column = st.columns(
            [3, 1.6],
            vertical_alignment="bottom",
        )
        heading.markdown("**学生综合排名**")
        sort_mode = sort_column.segmented_control(
            "排序方式",
            STUDENT_RANKING_SORT_OPTIONS,
            default=STUDENT_RANKING_SORT_OPTIONS[0],
            required=True,
            key=sort_key,
            width="stretch",
            persist_state="page",
        )
        heading.caption(
            "仅展示全部主题均证据充分的学生；综合得分率取各主题得分率的算术平均。"
        )
        qualified = [row for row in ranking if row["ranking_eligible"]]
        incomplete = sorted(
            (row for row in ranking if not row["ranking_eligible"]),
            key=lambda row: (
                -float(row["record_completeness"]),
                row["score_rate"] is None,
                -float(row["score_rate"] or 0),
                str(row["student_code"]),
            ),
        )
        visible = _sorted_ranking(qualified, str(sort_mode))
        table_key = f"analytics_student_ranking_table_{dataset_id}"
        special_table_key = (
            f"analytics_student_incomplete_ranking_table_{dataset_id}"
        )
        main_click_key = f"analytics_student_ranking_open_{dataset_id}"
        if visible:
            st.dataframe(
                _ranking_frame(visible),
                hide_index=True,
                width="stretch",
                height=315,
                key=table_key,
                column_config={
                    "综合名次": st.column_config.TextColumn(
                        "综合名次",
                        width="small",
                    ),
                    "综合得分率": st.column_config.NumberColumn(
                        "综合得分率",
                        format="percent",
                    ),
                    "已观察主题": st.column_config.TextColumn(
                        "已观察主题",
                        width="small",
                    ),
                    "记录完整度": st.column_config.TextColumn(
                        "记录完整度",
                        width="small",
                    ),
                    "查看": st.column_config.ButtonColumn(
                        "查看",
                        type="primary",
                        on_click=_open_student_detail,
                        kwargs={
                            "click_key": main_click_key,
                            "dataset_id": dataset_id,
                            "student_codes": [
                                str(row["student_code"]) for row in visible
                            ],
                        },
                        key=main_click_key,
                    ),
                },
                alt="记录完整且证据充分的学生综合排名列表",
            )
        else:
            st.info("当前没有记录完整且各主题证据充分的学生。")

        special_click_key = (
            f"analytics_student_incomplete_ranking_open_{dataset_id}"
        )
        if incomplete:
            special_heading, special_status = st.columns(
                [3, 1.5],
                vertical_alignment="center",
            )
            special_heading.markdown("**特殊榜单 · 记录不完整学生**")
            special_status.badge(
                f"{len(incomplete)} 人暂不参与综合排名",
                icon=":material/info:",
                color="orange",
            )
            st.caption(
                "记录完整度按“证据充分主题数 / 全部主题数”计算；"
                "特殊榜单单独排列，不影响综合名次。"
            )
            st.dataframe(
                _incomplete_ranking_frame(incomplete),
                hide_index=True,
                width="stretch",
                height=min(280, 50 + 35 * len(incomplete)),
                key=special_table_key,
                column_config={
                    "记录完整度": st.column_config.TextColumn(
                        "记录完整度",
                        width="small",
                    ),
                    "已观察主题": st.column_config.TextColumn(
                        "已观察主题",
                        width="small",
                    ),
                    "已观察主题得分率": st.column_config.NumberColumn(
                        "已观察主题得分率",
                        format="percent",
                    ),
                    "查看": st.column_config.ButtonColumn(
                        "查看",
                        type="primary",
                        on_click=_open_student_detail,
                        kwargs={
                            "click_key": special_click_key,
                            "dataset_id": dataset_id,
                            "student_codes": [
                                str(row["student_code"]) for row in incomplete
                            ],
                        },
                        key=special_click_key,
                    ),
                },
                alt="记录不完整、暂不参与综合排名的学生列表",
            )
        st.caption("点击对应榜单末列的“查看”，进入该学生的数据与分析页面。")


def _render_theme_overview(
    *,
    dataset_id: str,
    student_code: str,
    node_code: str | None,
    metrics: list[dict[str, Any]],
) -> None:
    scope_key = node_code or "all"
    with st.container(
        border=True,
        key=f"analytics_theme_overview_{dataset_id}_{student_code}_{scope_key}",
    ):
        st.subheader("主题概览", icon=":material/radar:")
        available = [
            metric
            for metric in metrics
            if metric.get("score_rate") is not None
        ]
        if available:
            st.echarts_chart(
                {
                    "tooltip": {
                        "trigger": "item",
                        "valueFormatter": "percent",
                    },
                    "radar": {
                        "indicator": [
                            {"name": metric["theme_name"], "max": 1}
                            for metric in available
                        ],
                        "radius": "62%",
                        "splitNumber": 4,
                    },
                    "series": [
                        {
                            "type": "radar",
                            "name": "主题得分率",
                            "data": [
                                {
                                    "name": "主题得分率",
                                    "value": [
                                        float(metric["score_rate"])
                                        for metric in available
                                    ],
                                }
                            ],
                        }
                    ],
                },
                height=370,
                key=f"analytics_student_radar_{dataset_id}_{student_code}_{scope_key}",
                alt=f"{student_code} 当前范围主题得分率概览",
            )
            missing = [
                metric["theme_name"]
                for metric in metrics
                if metric.get("score_rate") is None
            ]
            if missing:
                st.caption("信息不足、未纳入雷达图的主题：" + "、".join(missing))
        else:
            st.info("当前范围没有可绘制的主题表现。")
        st.dataframe(
            _metric_table(metrics),
            hide_index=True,
            width="stretch",
            alt="当前学生主题得分率与证据状态",
        )
        st.download_button(
            "下载主题统计 CSV",
            data=_csv_text(metrics),
            file_name=f"{student_code}_主题统计.csv",
            mime="text/csv",
            key=f"analytics_student_theme_csv_{dataset_id}_{student_code}_{scope_key}",
        )


def _render_indicator_overview(
    analytics: AnalyticsService,
    *,
    dataset_id: str,
    student_code: str,
    node_code: str | None,
) -> None:
    scope_key = node_code or "all"
    with st.container(
        border=True,
        key=f"analytics_indicator_overview_{dataset_id}_{student_code}_{scope_key}",
    ):
        st.subheader("观察指标", icon=":material/query_stats:")
        indicators = analytics.student_indicator_metrics(
            dataset_id,
            student_code,
            node_code,
        )
        available = [
            item for item in indicators if item.get("score_rate") is not None
        ]
        if available:
            chart_rows = pd.DataFrame(
                [
                    {
                        "观察指标": item["indicator_name"],
                        "得分率": float(item["score_rate"]) * 100,
                        "证据状态": EVIDENCE_LABELS[item["evidence_status"]],
                    }
                    for item in available
                ]
            )
            st.echarts_chart(
                {
                    "tooltip": {
                        "trigger": "axis",
                        "valueFormatter": "%.1f%%",
                    },
                    "grid": {
                        "left": 20,
                        "right": 36,
                        "top": 20,
                        "bottom": 20,
                        "containLabel": True,
                    },
                    "dataset": {"source": chart_rows},
                    "xAxis": {
                        "type": "value",
                        "min": 0,
                        "max": 100,
                        "axisLabel": {"formatter": "{value}%"},
                    },
                    "yAxis": {"type": "category"},
                    "series": [
                        {
                            "type": "bar",
                            "encode": {
                                "x": "得分率",
                                "y": "观察指标",
                            },
                            "barMaxWidth": 30,
                        }
                    ],
                },
                height=340,
                key=(
                    f"analytics_student_indicators_{dataset_id}_"
                    f"{student_code}_{scope_key}"
                ),
                alt=f"{student_code} 当前范围观察指标得分率",
            )
        else:
            st.info("当前范围没有可绘制的观察指标表现。")
        st.dataframe(
            _metric_table(indicators),
            hide_index=True,
            width="stretch",
            alt="当前学生观察指标得分率与证据状态",
        )


def _render_student_snapshots(
    analytics: AnalyticsService,
    *,
    dataset_id: str,
    student_code: str,
    node_code: str | None,
) -> None:
    scope_key = node_code or "all"
    with st.expander(
        "分析快照与导出",
        icon=":material/history:",
        key=(
            f"analytics_student_snapshots_{dataset_id}_"
            f"{student_code}_{scope_key}"
        ),
        on_change="rerun",
    ) as panel:
        if not panel.open:
            return
        snapshot_scope = {"student_code": student_code, "node_code": node_code}
        if st.button(
            "生成当前分析快照",
            key=(
                f"analytics_student_snapshot_create_{dataset_id}_"
                f"{student_code}_{scope_key}"
            ),
        ):
            snapshot = analytics.create_snapshot(dataset_id, snapshot_scope)
            st.success(f"已保存分析快照 {snapshot['id']}")
        snapshots = analytics.list_snapshots(dataset_id)
        matching_snapshots = [
            item for item in snapshots if item["scope"] == snapshot_scope
        ]
        if matching_snapshots:
            snapshot_id = st.selectbox(
                "与已保存快照比较",
                [item["id"] for item in matching_snapshots],
                key=(
                    f"analytics_student_snapshot_selector_{dataset_id}_"
                    f"{student_code}_{scope_key}"
                ),
            )
            if st.button(
                "查看前后变化",
                key=(
                    f"analytics_student_snapshot_compare_{dataset_id}_"
                    f"{student_code}_{scope_key}"
                ),
            ):
                comparison = analytics.compare_snapshot(snapshot_id)
                if comparison["changes"]:
                    st.dataframe(
                        comparison["changes"],
                        hide_index=True,
                        width="stretch",
                        alt="当前结果与所选分析快照的主题变化",
                    )
                else:
                    st.info("当前结果与所选快照没有主题表现或证据状态变化。")
                if comparison["affected_record_codes"]:
                    st.caption(
                        "新增或修改的作答依据："
                        + "、".join(comparison["affected_record_codes"])
                    )
        case_bundle = analytics.typical_case_bundle(
            dataset_id,
            student_code,
            node_code,
        )
        st.download_button(
            "下载当前学生典型案例 JSON",
            data=_json_text(case_bundle),
            file_name=f"{student_code}_典型案例.json",
            mime="application/json",
            key=(
                f"analytics_student_case_json_{dataset_id}_"
                f"{student_code}_{scope_key}"
            ),
        )


def _render_node_changes(
    analytics: AnalyticsService,
    *,
    dataset_id: str,
    student_code: str,
    metrics: list[dict[str, Any]],
    node_code: str | None,
) -> None:
    scope_key = node_code or "all"
    with st.container(
        border=True,
        key=f"analytics_node_changes_{dataset_id}_{student_code}_{scope_key}",
    ):
        st.subheader("节点变化", icon=":material/timeline:")
        if not metrics:
            st.info("当前学生还没有可用于比较的主题记录。")
            return
        theme_code = st.selectbox(
            "查看主题",
            [item["theme_code"] for item in metrics],
            format_func=lambda value: next(
                item["theme_name"]
                for item in metrics
                if item["theme_code"] == value
            ),
            key=(
                f"analytics_student_trend_theme_{dataset_id}_"
                f"{student_code}_{scope_key}"
            ),
        )
        trend = analytics.student_theme_trend(
            dataset_id,
            student_code,
            str(theme_code),
        )
        points = [
            point
            for point in trend["points"]
            if point.get("score_rate") is not None
        ]
        if points:
            trend_rows = pd.DataFrame(
                [
                    {
                        "节点": (
                            f"{point['node_code']} · "
                            f"{point.get('test_date') or '未标日期'}"
                        ),
                        "得分率": float(point["score_rate"]) * 100,
                        "证据状态": EVIDENCE_LABELS[
                            point["evidence_status"]
                        ],
                    }
                    for point in points
                ]
            )
            st.echarts_chart(
                {
                    "tooltip": {
                        "trigger": "axis",
                        "valueFormatter": "%.1f%%",
                    },
                    "grid": {
                        "left": 30,
                        "right": 25,
                        "top": 25,
                        "bottom": 45,
                        "containLabel": True,
                    },
                    "dataset": {"source": trend_rows},
                    "xAxis": {
                        "type": "category",
                        "axisLabel": {"rotate": 15},
                    },
                    "yAxis": {
                        "type": "value",
                        "min": 0,
                        "max": 100,
                        "axisLabel": {"formatter": "{value}%"},
                    },
                    "series": [
                        {
                            "type": "line",
                            "connectNulls": False,
                            "symbolSize": 8,
                            "encode": {"x": "节点", "y": "得分率"},
                        }
                    ],
                },
                height=330,
                key=(
                    f"analytics_student_trend_chart_{dataset_id}_"
                    f"{student_code}_{scope_key}_{theme_code}"
                ),
                alt=f"{student_code} 所选主题在各测验节点的得分率变化",
            )
            st.dataframe(
                trend_rows,
                hide_index=True,
                width="stretch",
                alt="所选主题在各测验节点的得分率与证据状态",
            )
        else:
            st.info("所选主题没有可绘制的节点表现。")
        if trend["status"] == "observable" and trend["delta"] is not None:
            st.caption(
                f"最近两个证据充分节点的得分率变化：{trend['delta']:+.1%}。"
            )
        elif trend["status"] != "observable":
            st.warning(
                "相邻节点证据不足，图中仅展示已有测验表现，不判定变化趋势。"
            )
        _render_student_snapshots(
            analytics,
            dataset_id=dataset_id,
            student_code=student_code,
            node_code=node_code,
        )


def _render_issue_evidence(
    issue: dict[str, Any],
    *,
    kind: str,
    scope_key: str,
) -> None:
    theme_code = str(issue["theme_code"])
    with st.expander(
        "依据来源与对应记录",
        icon=":material/source:",
        key=f"analytics_issue_evidence_{kind}_{scope_key}_{theme_code}",
        on_change="rerun",
    ) as evidence_panel:
        if not evidence_panel.open:
            return
        st.markdown("**依据来源**")
        source_rows = issue.get("sources", [])
        if source_rows:
            st.dataframe(
                source_rows,
                hide_index=True,
                width="stretch",
                alt=f"{issue['theme_name']}的课标、教材与指标依据",
            )
        else:
            st.caption("当前主题没有关联的课程要求来源。")
        st.markdown("**对应作答记录**")
        record_rows = issue.get("records", [])
        if record_rows:
            st.dataframe(
                record_rows,
                hide_index=True,
                width="stretch",
                alt=f"{issue['theme_name']}对应的有效评分记录",
            )
        else:
            st.caption("当前范围没有可追溯的有效评分记录。")


def _render_issue_card(
    issue: dict[str, Any],
    *,
    kind: str,
    scope_key: str,
) -> None:
    theme_code = str(issue["theme_code"])
    with st.container(
        border=True,
        key=f"analytics_issue_{kind}_{scope_key}_{theme_code}",
    ):
        title, rate, comparison = st.columns([3, 1.2, 1.5])
        title.markdown(f"**{issue['theme_name']}**")
        title.caption(issue["reason"])
        rate.metric(
            "当前得分率",
            f"{float(issue['score_rate']):.1%}",
        )
        if kind == "personalized":
            comparison.metric(
                "较班级平均",
                f"{float(issue['delta']):+.1%}",
                border=False,
            )
            comparison.caption(
                f"班级平均 {float(issue['class_score_rate']):.1%}"
            )
        else:
            comparison.metric("证据状态", "证据充分")
            comparison.caption("已达到巩固判定门槛")
        _render_issue_evidence(
            issue,
            kind=kind,
            scope_key=scope_key,
        )


def _render_problem_analysis(
    analytics: AnalyticsService,
    diagnosis: StudentDiagnosisService,
    profile: ProviderProfile | None,
    *,
    dataset_id: str,
    student_code: str,
    node_code: str | None,
) -> None:
    analysis = analytics.student_issue_analysis(
        dataset_id,
        student_code,
        node_code,
    )
    scope_key = f"{dataset_id}_{student_code}_{node_code or 'all'}"
    summary = st.columns(3)
    summary[0].metric(
        "需要巩固",
        len(analysis["need_consolidation"]),
        border=True,
    )
    summary[1].metric(
        "个性化问题",
        len(analysis["personalized"]),
        border=True,
    )
    summary[2].metric(
        "待补充观察",
        len(analysis["supplemental"]),
        border=True,
    )
    st.caption(
        f"判定阈值：证据充分且得分率低于 {analysis['threshold']:.0%}；"
        "证据不足、待评分和未作答不判为问题。"
    )

    st.subheader("需要巩固的内容", icon=":material/priority_high:")
    need_consolidation = sorted(
        analysis["need_consolidation"],
        key=lambda item: float(item["score_rate"]),
    )
    if need_consolidation:
        for issue in need_consolidation:
            _render_issue_card(
                issue,
                kind="consolidation",
                scope_key=scope_key,
            )
    else:
        st.success("当前没有达到证据门槛且低于巩固阈值的主题。")

    st.subheader("个性化问题", icon=":material/person_alert:")
    personalized = sorted(
        analysis["personalized"],
        key=lambda item: float(item["delta"]),
    )
    if personalized:
        for issue in personalized:
            _render_issue_card(
                issue,
                kind="personalized",
                scope_key=scope_key,
            )
    else:
        st.success("当前学生在已充分观察的主题中没有低于同班平均的表现。")

    st.subheader("待补充观察", icon=":material/visibility:")
    if analysis["supplemental"]:
        st.caption(
            "以下主题证据不足，只建议补充观察，不形成掌握或未掌握结论。"
        )
        st.dataframe(
            [
                {
                    "主题": item["theme_name"],
                    "证据状态": EVIDENCE_LABELS[item["evidence_status"]],
                    "有效评分": item["scored_records"],
                    "待评分": item["pending_records"],
                    "未作答": item["unanswered_records"],
                    "说明": item["reason"],
                }
                for item in analysis["supplemental"]
            ],
            hide_index=True,
            width="stretch",
            alt=f"{student_code} 证据不足、需要补充观察的主题",
        )
        suggestions = analytics.supplemental_observations(
            dataset_id,
            student_code,
            node_code or "",
        ) if node_code is not None else []
        if suggestions:
            st.dataframe(
                suggestions,
                hide_index=True,
                width="stretch",
                alt="可补充观察的已有题目与材料",
            )
    else:
        st.success("当前学生的已导入主题均达到最小有效记录门槛。")

    with st.expander(
        "AI 诊断与教师复核",
        icon=":material/clinical_notes:",
        key=f"analytics_ai_diagnosis_{scope_key}",
        on_change="rerun",
    ) as diagnosis_panel:
        if not diagnosis_panel.open:
            return
        _render_ai_diagnosis_tab(
            diagnosis,
            profile,
            dataset_id=dataset_id,
            student_code=student_code,
            class_code=analysis["class_code"],
            node_code=node_code,
        )


def render_student_analysis_page(
    analytics: AnalyticsService,
    learning: LearningDataService,
    diagnosis: StudentDiagnosisService,
    profile: ProviderProfile | None,
) -> None:
    dataset = _render_student_dataset_context(learning)
    if dataset is None:
        return
    ranking = analytics.student_performance_ranking(dataset["id"])
    if not ranking:
        st.info("该数据集尚未导入学生记录。")
        return
    selected_code = st.session_state.get(STUDENT_DETAIL_STATE_KEY)
    student_codes = {
        str(row["student_code"]) for row in ranking
    }
    if selected_code not in student_codes:
        st.session_state.pop(STUDENT_DETAIL_STATE_KEY, None)
        _render_student_rankings(dataset["id"], ranking)
        return
    selected = next(
        row
        for row in ranking
        if str(row["student_code"]) == str(selected_code)
    )
    student_code = str(selected["student_code"])
    section = _render_student_analysis_top_nav()
    if selected["ranking_eligible"]:
        st.caption(
            f"学生 {student_code} · {selected['class_code']} · "
            f"综合名次第 {selected['rank']} 名 · "
            f"综合得分率 {selected['score_rate']:.1%}"
        )
    else:
        st.caption(
            f"学生 {student_code} · {selected['class_code']} · "
            f"暂不参与综合排名 · 记录完整度 "
            f"{selected['record_completeness']:.0%}"
        )
    node_code = _node_selector(
        learning,
        dataset["id"],
        key=f"analytics_student_node_{dataset['id']}",
    )
    metrics = analytics.student_theme_metrics(
        dataset["id"],
        student_code,
        node_code,
    )

    if section == "问题分析":
        _render_problem_analysis(
            analytics,
            diagnosis,
            profile,
            dataset_id=dataset["id"],
            student_code=student_code,
            node_code=node_code,
        )
        return

    observed_metrics = [
        metric for metric in metrics if metric.get("score_rate") is not None
    ]
    scope_score_rate = (
        sum(float(metric["score_rate"]) for metric in observed_metrics)
        / len(observed_metrics)
        if observed_metrics
        else None
    )
    st.subheader("数据统计标签", icon=":material/label:")
    with st.grid("auto", min_column_width=150, gap="small"):
        st.metric(
            "当前范围综合得分率",
            (
                f"{scope_score_rate:.1%}"
                if scope_score_rate is not None
                else "—"
            ),
            border=True,
            height="stretch",
        )
        st.metric(
            "综合名次",
            (
                f"第 {selected['rank']} 名"
                if selected["ranking_eligible"]
                else "不参与排名"
                if selected["score_rate"] is not None
                else "暂无排名"
            ),
            border=True,
            height="stretch",
        )
        st.metric(
            "有效评分",
            sum(int(item["scored_records"]) for item in metrics),
            border=True,
            height="stretch",
        )
        st.metric(
            "待评分/未作答",
            sum(
                int(item["pending_records"]) + int(item["unanswered_records"])
                for item in metrics
            ),
            border=True,
            height="stretch",
        )
        st.metric(
            "需补充观察",
            sum(
                item["evidence_status"] != "sufficient"
                for item in metrics
            ),
            border=True,
            height="stretch",
        )
    if not metrics:
        st.info("所选学生和节点没有作答记录。")
        return
    _render_theme_overview(
        dataset_id=dataset["id"],
        student_code=student_code,
        node_code=node_code,
        metrics=metrics,
    )
    _render_indicator_overview(
        analytics,
        dataset_id=dataset["id"],
        student_code=student_code,
        node_code=node_code,
    )
    _render_node_changes(
        analytics,
        dataset_id=dataset["id"],
        student_code=student_code,
        metrics=metrics,
        node_code=node_code,
    )


def _render_class_comparison(
    analytics: AnalyticsService,
    *,
    dataset_id: str,
    node_code: str | None,
    metrics: list[dict[str, Any]],
) -> None:
    summary = st.columns(4)
    summary[0].metric(
        "班级数",
        len({item["class_code"] for item in metrics}),
        border=True,
    )
    summary[1].metric(
        "主题数",
        len({item["theme_code"] for item in metrics}),
        border=True,
    )
    summary[2].metric(
        "证据充分",
        sum(item["evidence_status"] == "sufficient" for item in metrics),
        border=True,
    )
    summary[3].metric(
        "待补充观察",
        sum(item["evidence_status"] != "sufficient" for item in metrics),
        border=True,
    )
    figure = go.Figure()
    for class_code in sorted({item["class_code"] for item in metrics}):
        rows = [item for item in metrics if item["class_code"] == class_code]
        figure.add_trace(
            go.Bar(
                name=class_code,
                x=[item["theme_name"] for item in rows],
                y=[(item["score_rate"] or 0) * 100 for item in rows],
                customdata=[
                    [
                        item["student_count"],
                        item["scored_records"],
                        EVIDENCE_LABELS[item["evidence_status"]],
                    ]
                    for item in rows
                ],
                hovertemplate=(
                    "%{x}<br>得分率：%{y:.1f}%<br>学生数：%{customdata[0]}"
                    "<br>有效记录：%{customdata[1]}<br>%{customdata[2]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        barmode="group",
        yaxis={"range": [0, 100], "title": "得分率（%）"},
        margin={"l": 30, "r": 20, "t": 30, "b": 30},
    )
    st.plotly_chart(
        figure,
        width="stretch",
        key="analytics_class_comparison",
        alt="各班级在不同主题上的平均得分率对比",
    )
    st.download_button(
        "下载班级主题统计 CSV",
        data=_csv_text(metrics),
        file_name="班级主题统计.csv",
        mime="text/csv",
        key="analytics_class_theme_csv",
    )
    distribution = analytics.class_theme_distribution(dataset_id, node_code)
    st.subheader("学生主题表现分布")
    if distribution:
        distribution_figure = go.Figure()
        for item in distribution:
            distribution_figure.add_trace(
                go.Box(
                    y=[rate * 100 for rate in item["student_rates"]],
                    name=(
                        f"{item['class_code']} · {item['theme_name']}"
                        f"（{item['observed_students']}人）"
                    ),
                    boxpoints="all",
                    hovertemplate="已观察得分率：%{y:.1f}%<extra></extra>",
                )
            )
        distribution_figure.update_layout(
            yaxis={"range": [0, 100], "title": "已观察得分率（%）"},
            margin={"l": 30, "r": 20, "t": 30, "b": 90},
        )
        st.plotly_chart(
            distribution_figure,
            width="stretch",
            key="analytics_class_distribution",
            alt="各班级学生主题得分率分布",
        )
        st.caption("仅纳入主题有效记录达到 3 条的学生；未纳入不表示未掌握。")
    else:
        st.info("没有达到最小有效记录门槛的学生主题数据，暂不展示分布。")
    st.subheader("主题数据充分度")
    heatmap = go.Figure(
        go.Heatmap(
            x=sorted({item["theme_name"] for item in metrics}),
            y=sorted({item["class_code"] for item in metrics}),
            z=[
                [
                    next(
                        (
                            item["scored_records"]
                            for item in metrics
                            if item["class_code"] == class_code
                            and item["theme_name"] == theme_name
                        ),
                        0,
                    )
                    for theme_name in sorted({item["theme_name"] for item in metrics})
                ]
                for class_code in sorted({item["class_code"] for item in metrics})
            ],
            colorscale="Blues",
            hovertemplate="班级：%{y}<br>主题：%{x}<br>有效记录：%{z}<extra></extra>",
        )
    )
    heatmap.update_layout(margin={"l": 30, "r": 20, "t": 30, "b": 30})
    st.plotly_chart(
        heatmap,
        width="stretch",
        key="analytics_class_coverage",
        alt="各班级各主题有效记录数量热力图",
    )
    common_observations = sorted(
        (
            item
            for item in metrics
            if item["evidence_status"] == "sufficient"
            and item["score_rate"] is not None
        ),
        key=lambda item: float(item["score_rate"]),
    )[:3]
    st.subheader("优先补充观察的共同主题")
    if common_observations:
        st.dataframe(
            [
                {
                    "班级": item["class_code"],
                    "主题": item["theme_name"],
                    "已观察得分率": f"{item['score_rate']:.1%}",
                    "有效记录": item["scored_records"],
                    "说明": "用于安排下一步课堂观察，不等同于掌握结论。",
                }
                for item in common_observations
            ],
            hide_index=True,
            width="stretch",
            alt="需要优先关注的低得分率班级主题",
        )
    else:
        st.info("暂无证据充分的共同主题可排序；请先补充有效作答。")
    st.dataframe(
        _metric_table(metrics),
        hide_index=True,
        width="stretch",
        alt="班级主题得分率与证据状态统计",
    )


def _render_class_issue_card(
    issue: dict[str, Any],
    *,
    dataset_id: str,
    node_code: str | None,
    threshold: float,
) -> None:
    scope_key = (
        f"{dataset_id}_{issue['class_code']}_{node_code or 'all'}"
    )
    theme_code = str(issue["theme_code"])
    with st.container(
        border=True,
        key=f"class_issue_common_{scope_key}_{theme_code}",
    ):
        heading, status = st.columns(
            [5, 1.2],
            vertical_alignment="center",
        )
        heading.markdown(
            f":material/warning: **{issue['theme_name']}**"
        )
        heading.caption(issue["reason"])
        status.badge(
            "需要巩固",
            icon=":material/priority_high:",
            color="red",
        )
        signals = st.columns(4)
        signals[0].metric(
            "班级平均得分率",
            f"{float(issue['score_rate']):.1%}",
        )
        signals[1].metric(
            "有效观察学生",
            f"{int(issue['student_count'])} 人",
        )
        signals[2].metric(
            "有效评分记录",
            f"{int(issue['scored_records'])} 条",
        )
        signals[3].metric(
            "低于巩固阈值",
            f"{float(issue['threshold_gap']):.1%}",
        )
        st.progress(
            float(issue["score_rate"]),
            text=(
                f"得分率 {float(issue['score_rate']):.1%}"
                f" / 判定阈值 {threshold:.0%}"
            ),
        )
        _render_issue_evidence(
            issue,
            kind="class_common",
            scope_key=scope_key,
        )


def _render_specific_class_analysis(
    analytics: AnalyticsService,
    *,
    dataset_id: str,
    node_code: str | None,
    metrics: list[dict[str, Any]],
) -> None:
    class_codes = sorted({str(item["class_code"]) for item in metrics})
    class_code = st.selectbox(
        "具体班级",
        class_codes,
        key=f"analytics_specific_class_{dataset_id}_{node_code or 'all'}",
    )
    analysis = analytics.class_issue_analysis(
        dataset_id,
        str(class_code),
        node_code,
    )
    summary = st.grid("auto", min_column_width=150, gap="small")
    summary.metric(
        "班级学生数",
        analysis["student_count"],
        border=True,
        height="stretch",
    )
    summary.metric(
        "班级综合得分率",
        (
            f"{analysis['composite_score_rate']:.1%}"
            if analysis["composite_score_rate"] is not None
            else "—"
        ),
        border=True,
        height="stretch",
        help=(
            f"已观察 {analysis['observed_theme_count']}/"
            f"{analysis['total_theme_count']} 个主题"
        ),
    )
    summary.metric(
        "共性问题",
        len(analysis["common"]),
        border=True,
        height="stretch",
        help="证据充分且班级平均得分率低于巩固阈值的主题数",
    )
    summary.metric(
        "待补充观察",
        len(analysis["supplemental"]),
        border=True,
        height="stretch",
        help="尚未达到班级共性问题判定门槛的主题数",
    )
    st.caption(
        f"判定阈值：证据充分且班级平均得分率低于 "
        f"{analysis['threshold']:.0%}；证据不足不判为共性问题。"
    )

    st.subheader("共性问题", icon=":material/groups:")
    common = sorted(
        analysis["common"],
        key=lambda item: float(item["score_rate"]),
    )
    if common:
        st.caption(
            "以下主题同时显示问题得分、覆盖学生、有效记录与阈值差距，"
            "并可按主题追溯课标、教材和作答记录。"
        )
        for issue in common:
            _render_class_issue_card(
                issue,
                dataset_id=dataset_id,
                node_code=node_code,
                threshold=float(analysis["threshold"]),
            )
    else:
        st.success("当前班级在已充分观察的主题中没有共性问题。")

    st.subheader("待补充观察", icon=":material/visibility:")
    if analysis["supplemental"]:
        st.dataframe(
            [
                {
                    "主题": item["theme_name"],
                    "证据状态": EVIDENCE_LABELS[item["evidence_status"]],
                    "已观察学生": f"{item['student_count']} 人",
                    "有效评分": item["scored_records"],
                    "待评分": item["pending_records"],
                    "未作答": item["unanswered_records"],
                    "说明": item["reason"],
                }
                for item in analysis["supplemental"]
            ],
            hide_index=True,
            width="stretch",
            alt=f"{class_code} 尚未达到共性问题判定门槛的主题",
        )
    else:
        st.success("当前班级的已导入主题均达到共性问题判定门槛。")


def render_class_analysis_page(
    analytics: AnalyticsService,
    learning: LearningDataService,
) -> None:
    section = _render_class_analysis_top_nav()
    dataset = _dataset_selector(learning, key="analytics_class_dataset")
    if dataset is None:
        return
    node_code = _node_selector(learning, dataset["id"], key="analytics_class_node")
    metrics = analytics.class_theme_metrics(dataset["id"], node_code)
    if not metrics:
        st.info("所选范围没有可比较的班级作答记录。")
        return
    if section == "具体班级分析":
        _render_specific_class_analysis(
            analytics,
            dataset_id=dataset["id"],
            node_code=node_code,
            metrics=metrics,
        )
        return
    _render_class_comparison(
        analytics,
        dataset_id=dataset["id"],
        node_code=node_code,
        metrics=metrics,
    )
