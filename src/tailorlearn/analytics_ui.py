from __future__ import annotations

import csv
from io import StringIO
import json
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from tailorlearn.analytics import AnalyticsService
from tailorlearn.ai import AIRequestError, OpenAICompatibleClient
from tailorlearn.learning_data import LearningDataService
from tailorlearn.models import ProviderProfile
from tailorlearn.student_diagnosis import StudentDiagnosisService


EVIDENCE_LABELS = {
    "sufficient": "证据充分",
    "limited": "证据有限",
    "insufficient": "信息不足",
    "unavailable": "不可用",
}
EVIDENCE_COLORS = {
    "sufficient": "#1f77b4",
    "limited": "#ff9f1c",
    "insufficient": "#9aa0a6",
    "unavailable": "#d62728",
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
        st.info("尚无学情数据集。请先在“材料中心”导入数据。")
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


def _render_radar(metrics: list[dict[str, Any]]) -> None:
    available = [metric for metric in metrics if metric.get("score_rate") is not None]
    if not available:
        st.info("所选范围没有可绘制的主题表现；请查看数据充分度和待评分记录。")
        return
    labels = [metric["theme_name"] for metric in available]
    rates = [float(metric["score_rate"]) * 100 for metric in available]
    figure = go.Figure(
        go.Scatterpolar(
            r=[*rates, rates[0]],
            theta=[*labels, labels[0]],
            fill="toself",
            name="已观察表现",
            customdata=[
                [
                    metric["scored_records"],
                    EVIDENCE_LABELS[metric["evidence_status"]],
                ]
                for metric in [*available, available[0]]
            ],
            hovertemplate=(
                "%{theta}<br>得分率：%{r:.1f}%<br>有效记录：%{customdata[0]}"
                "<br>%{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch", key="analytics_student_radar")
    missing = [
        metric["theme_name"]
        for metric in metrics
        if metric.get("score_rate") is None
    ]
    if missing:
        st.caption("未绘制的信息不足主题：" + "、".join(missing))


def _render_trend(analytics: AnalyticsService, dataset_id: str, student: str, theme: str) -> None:
    trend = analytics.student_theme_trend(dataset_id, student, theme)
    points = [point for point in trend["points"] if point.get("score_rate") is not None]
    if not points:
        st.info("没有可绘制的节点表现。")
        return
    figure = go.Figure(
        go.Scatter(
            x=[point["test_date"] or point["node_code"] for point in points],
            y=[point["score_rate"] * 100 for point in points],
            mode="lines+markers",
            text=[
                f"{point['node_code']} · {EVIDENCE_LABELS[point['evidence_status']]}"
                for point in points
            ],
            hovertemplate="%{text}<br>得分率：%{y:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        yaxis={"range": [0, 100], "title": "得分率（%）"},
        xaxis={"title": "测验节点"},
        margin={"l": 30, "r": 20, "t": 30, "b": 30},
    )
    st.plotly_chart(figure, width="stretch", key="analytics_student_trend")
    if trend["status"] != "observable":
        st.warning("前驱或后继节点证据不足，图中仅展示已有测验表现，不判定变化趋势。")
    elif trend["delta"] is not None:
        st.caption(f"相邻充分证据节点的得分率变化：{trend['delta']:+.1%}。")


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


def render_student_analysis_page(
    analytics: AnalyticsService,
    learning: LearningDataService,
    diagnosis: StudentDiagnosisService,
    profile: ProviderProfile | None,
) -> None:
    dataset = _dataset_selector(learning, key="analytics_student_dataset")
    if dataset is None:
        return
    students = learning.list_records(dataset["id"], "student")
    if not students:
        st.info("该数据集尚未导入学生记录。")
        return
    student_code = st.selectbox(
        "学生",
        [row["student_code"] for row in students],
        key="analytics_student_selector",
    )
    class_code = next(
        str(row.get("class_code") or "未分班")
        for row in students
        if row["student_code"] == student_code
    )
    node_code = _node_selector(
        learning,
        dataset["id"],
        key="analytics_student_node",
    )
    metrics = analytics.student_theme_metrics(dataset["id"], student_code, node_code)
    if not metrics:
        st.info("所选学生和节点没有作答记录。")
        return
    summary = st.columns(4)
    summary[0].metric("主题数", len(metrics), border=True)
    summary[1].metric(
        "有效记录",
        sum(item["scored_records"] for item in metrics),
        border=True,
    )
    summary[2].metric(
        "待评分/未作答",
        sum(item["pending_records"] + item["unanswered_records"] for item in metrics),
        border=True,
    )
    summary[3].metric(
        "需补充观察",
        sum(item["evidence_status"] != "sufficient" for item in metrics),
        border=True,
    )
    radar_tab, indicator_tab, trend_tab, evidence_tab, diagnosis_tab = st.tabs(
        ["主题概览", "观察指标", "节点变化", "依据与补充观察", "AI 诊断与教师复核"]
    )
    with radar_tab:
        _render_radar(metrics)
        st.dataframe(_metric_table(metrics), hide_index=True, width="stretch")
        st.download_button(
            "下载主题统计 CSV",
            data=_csv_text(metrics),
            file_name=f"{student_code}_主题统计.csv",
            mime="text/csv",
            key="analytics_student_theme_csv",
        )
    with indicator_tab:
        indicators = analytics.student_indicator_metrics(
            dataset["id"], student_code, node_code
        )
        figure = go.Figure(
            go.Bar(
                x=[(item["score_rate"] or 0) * 100 for item in indicators],
                y=[item["indicator_name"] for item in indicators],
                orientation="h",
                marker_color=[EVIDENCE_COLORS[item["evidence_status"]] for item in indicators],
                customdata=[
                    [item["scored_records"], EVIDENCE_LABELS[item["evidence_status"]]]
                    for item in indicators
                ],
                hovertemplate=(
                    "%{y}<br>得分率：%{x:.1f}%<br>有效记录：%{customdata[0]}"
                    "<br>%{customdata[1]}<extra></extra>"
                ),
            )
        )
        figure.update_layout(
            xaxis={"range": [0, 100], "title": "得分率（%）"},
            margin={"l": 30, "r": 20, "t": 30, "b": 30},
        )
        st.plotly_chart(figure, width="stretch", key="analytics_student_indicators")
        st.dataframe(_metric_table(indicators), hide_index=True, width="stretch")
    with trend_tab:
        theme_code = st.selectbox(
            "主题",
            [item["theme_code"] for item in metrics],
            format_func=lambda value: next(
                item["theme_name"] for item in metrics if item["theme_code"] == value
            ),
            key="analytics_student_trend_theme",
        )
        _render_trend(analytics, dataset["id"], student_code, theme_code)
    with evidence_tab:
        st.subheader("分析快照与变化")
        snapshot_scope = {"student_code": student_code, "node_code": node_code}
        if st.button("生成当前分析快照", key="analytics_student_snapshot_create"):
            snapshot = analytics.create_snapshot(dataset["id"], snapshot_scope)
            st.success(f"已保存分析快照 {snapshot['id']}")
        snapshots = analytics.list_snapshots(dataset["id"])
        matching_snapshots = [
            item for item in snapshots if item["scope"] == snapshot_scope
        ]
        if matching_snapshots:
            snapshot_id = st.selectbox(
                "与已保存快照比较",
                [item["id"] for item in matching_snapshots],
                key="analytics_student_snapshot_selector",
            )
            if st.button("查看前后变化", key="analytics_student_snapshot_compare"):
                comparison = analytics.compare_snapshot(snapshot_id)
                if comparison["changes"]:
                    st.dataframe(comparison["changes"], hide_index=True, width="stretch")
                else:
                    st.info("当前结果与所选快照没有主题表现或证据状态变化。")
                if comparison["affected_record_codes"]:
                    st.caption(
                        "新增或修改的作答依据："
                        + "、".join(comparison["affected_record_codes"])
                    )
        case_bundle = analytics.typical_case_bundle(
            dataset["id"], student_code, node_code
        )
        st.download_button(
            "下载当前学生典型案例 JSON",
            data=_json_text(case_bundle),
            file_name=f"{student_code}_典型案例.json",
            mime="application/json",
            key="analytics_student_case_json",
        )
        if node_code is None:
            st.info("请选择一个具体测验节点，以查看可回溯的作答证据和补充观察。")
            return
        theme_code = st.selectbox(
            "查看主题依据",
            [item["theme_code"] for item in metrics],
            format_func=lambda value: next(
                item["theme_name"] for item in metrics if item["theme_code"] == value
            ),
            key="analytics_student_evidence_theme",
        )
        evidence = analytics.theme_evidence(
            dataset["id"], student_code, node_code, theme_code
        )
        st.caption("观察指标：" + "、".join(evidence["requirement_ids"]))
        st.dataframe(evidence["records"], hide_index=True, width="stretch")
        suggestions = analytics.supplemental_observations(
            dataset["id"], student_code, node_code
        )
        if suggestions:
            st.subheader("建议补充观察")
            st.dataframe(suggestions, hide_index=True, width="stretch")
        else:
            st.success("当前指标均达到最小有效记录门槛。")
    with diagnosis_tab:
        _render_ai_diagnosis_tab(
            diagnosis,
            profile,
            dataset_id=dataset["id"],
            student_code=student_code,
            class_code=class_code,
            node_code=node_code,
        )


def render_class_analysis_page(
    analytics: AnalyticsService,
    learning: LearningDataService,
) -> None:
    dataset = _dataset_selector(learning, key="analytics_class_dataset")
    if dataset is None:
        return
    node_code = _node_selector(learning, dataset["id"], key="analytics_class_node")
    metrics = analytics.class_theme_metrics(dataset["id"], node_code)
    if not metrics:
        st.info("所选范围没有可比较的班级作答记录。")
        return
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
    st.plotly_chart(figure, width="stretch", key="analytics_class_comparison")
    st.download_button(
        "下载班级主题统计 CSV",
        data=_csv_text(metrics),
        file_name="班级主题统计.csv",
        mime="text/csv",
        key="analytics_class_theme_csv",
    )
    distribution = analytics.class_theme_distribution(dataset["id"], node_code)
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
    st.plotly_chart(heatmap, width="stretch", key="analytics_class_coverage")
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
        )
    else:
        st.info("暂无证据充分的共同主题可排序；请先补充有效作答。")
    st.dataframe(_metric_table(metrics), hide_index=True, width="stretch")
