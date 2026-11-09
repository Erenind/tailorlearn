from __future__ import annotations

import csv
from io import StringIO
import json
from typing import Any

import streamlit as st

from tailorlearn.ai import AIRequestError, OpenAICompatibleClient
from tailorlearn.learning_data import LearningDataService
from tailorlearn.models import ProviderProfile
from tailorlearn.teaching_recommendations import TeachingRecommendationService


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + output.getvalue()


def recommendation_csv_sections(bundle: dict[str, Any]) -> dict[str, str]:
    evidence = bundle["evidence_pack"]
    return {
        "班级主题统计.csv": _csv_text(evidence["class_themes"]),
        "测验蓝图.csv": _csv_text(evidence["assessment_blueprint"]),
        "教学分层.csv": _csv_text(evidence["groups"]),
    }


def recommendation_markdown(bundle: dict[str, Any]) -> str:
    evidence = bundle["evidence_pack"]
    lines = [
        "# 班级教学建议案例",
        "",
        f"- 班级：{bundle['class_code']}",
        f"- 测验节点：{bundle['node_code']}",
        "- 原则：以下为已观察表现与待补充观察，不等同于掌握判定。",
        "",
        "## 班级主题统计",
    ]
    for item in evidence["class_themes"]:
        rate = item.get("score_rate")
        rate_text = f"{rate:.1%}" if rate is not None else "—"
        lines.append(
            f"- {item['theme_name']}：{rate_text}；{item['evidence_status']}；"
            f"有效记录 {item['scored_records']}。"
        )
    lines.extend(["", "## 测验蓝图"])
    for item in evidence["assessment_blueprint"]:
        lines.append(
            f"- {item['theme_code']} / {item['indicator_code']}："
            f"{item['coverage_status']}；有效 {item['valid_answer_count']}；"
            f"待评分 {item['pending_count']}；未作答 {item['unanswered_count']}。"
        )
    lines.extend(["", "## 分层观察"])
    for item in evidence["groups"]:
        lines.append(
            f"- {item['theme_code']} / {item['group']}："
            f"{', '.join(item['student_codes']) or '无'}。{item['reason']}"
        )
    recommendation = bundle.get("recommendation")
    if recommendation:
        lines.extend(["", "## AI 草稿与教师审核"])
        lines.append(f"- 状态：{recommendation['status']}")
        lines.append(f"- 草稿摘要：{recommendation['draft']['summary']}")
        if recommendation["reviewed_summary"]:
            lines.append(f"- 教师修订：{recommendation['reviewed_summary']}")
        if recommendation["review_note"]:
            lines.append(f"- 审核备注：{recommendation['review_note']}")
    return "\n".join(lines) + "\n"


def _dataset_selector(learning: LearningDataService) -> dict[str, Any] | None:
    datasets = learning.list_datasets()
    if not datasets:
        st.info("尚无学情数据集。请先在“学情数据”页面导入数据。")
        return None
    dataset_id = st.selectbox(
        "学情数据集",
        [item["id"] for item in datasets],
        key="teaching_dataset",
        format_func=lambda value: next(
            (item["name"] for item in datasets if item["id"] == value), value
        ),
    )
    return next(item for item in datasets if item["id"] == dataset_id)


def _scope_selector(
    learning: LearningDataService,
    dataset_id: str,
) -> tuple[str, str] | None:
    students = learning.list_records(dataset_id, "student")
    class_codes = sorted({str(item.get("class_code") or "未分班") for item in students})
    nodes = learning.list_records(dataset_id, "node")
    if not class_codes or not nodes:
        st.info("需要先导入学生班级和至少一个测验节点。")
        return None
    columns = st.columns(2)
    class_code = columns[0].selectbox("班级", class_codes, key="teaching_class")
    node_code = columns[1].selectbox(
        "测验节点",
        [str(item["node_code"]) for item in nodes],
        key="teaching_node",
        format_func=lambda value: next(
            (
                f"{item['node_code']} · {item.get('test_date') or '未标日期'}"
                for item in nodes
                if item["node_code"] == value
            ),
            value,
        ),
    )
    return class_code, node_code


def _show_draft(draft: dict[str, Any]) -> None:
    st.subheader("AI 教学建议草稿")
    st.write(draft["summary"])
    for label, key in (
        ("已观察到的表现", "observations"),
        ("信息缺口", "information_gaps"),
        ("分层教学活动", "group_activities"),
        ("材料建议", "material_suggestions"),
    ):
        if draft.get(key):
            st.markdown(f"**{label}**")
            st.dataframe(draft[key], hide_index=True, width="stretch")
    if draft.get("teacher_actions"):
        st.markdown("**教师下一步行动**")
        st.write(draft["teacher_actions"])
    if draft.get("post_lesson_checks"):
        st.markdown("**课后检查**")
        st.write(draft["post_lesson_checks"])


def render_teaching_recommendations_page(
    service: TeachingRecommendationService,
    learning: LearningDataService,
    profile: ProviderProfile | None,
) -> None:
    st.title("教学建议")
    st.caption("AI 仅解释已核验的班级证据；信息不足时只提出补充观察，不判定未掌握。")
    dataset = _dataset_selector(learning)
    if dataset is None:
        return
    scope = _scope_selector(learning, dataset["id"])
    if scope is None:
        return
    class_code, node_code = scope
    service.refresh_stale_recommendations(dataset["id"])
    try:
        evidence = service.build_evidence_pack(dataset["id"], class_code, node_code)
    except ValueError as exc:
        st.info(str(exc))
        return
    st.subheader("班级主题证据")
    if evidence["class_themes"]:
        st.dataframe(evidence["class_themes"], hide_index=True, width="stretch")
    else:
        st.info("当前范围没有可比较的班级主题记录。")
    st.subheader("测验蓝图与覆盖缺口")
    st.dataframe(evidence["assessment_blueprint"], hide_index=True, width="stretch")
    st.subheader("分层观察")
    if evidence["groups"]:
        st.dataframe(evidence["groups"], hide_index=True, width="stretch")
    else:
        st.info("暂无可分层的主题；请先补充有效作答。")

    latest = service.get_latest_recommendation(dataset["id"], class_code, node_code)
    if profile is None:
        st.info("请先在“AI 参数设置”保存并启用厂商配置；仍可查看证据和历史草稿。")
    elif st.button("生成教学建议草稿", key="teaching_generate"):
        try:
            with st.spinner("正在生成可审核的教学建议草稿..."):
                latest = service.create_recommendation(
                    dataset["id"],
                    class_code,
                    node_code,
                    profile,
                    OpenAICompatibleClient(profile),
                )
            st.success("教学建议草稿已生成，等待教师复核。")
        except (AIRequestError, ValueError) as exc:
            st.error(str(exc))
    if latest:
        if latest["status"] == "stale":
            st.warning("当前草稿对应的数据已更新，请重新生成并审核。")
        _show_draft(latest["draft"])
        reviewed_summary = st.text_area(
            "教师修订摘要",
            value=latest["reviewed_summary"] or latest["draft"]["summary"],
            key=f"teaching_summary_{latest['id']}",
        )
        review_note = st.text_area(
            "审核备注",
            value=latest["review_note"] or "",
            key=f"teaching_note_{latest['id']}",
        )
        accept, reject = st.columns(2)
        if accept.button("采纳教师修订", key=f"teaching_accept_{latest['id']}"):
            service.review_recommendation(
                latest["id"],
                status="accepted",
                reviewed_summary=reviewed_summary,
                review_note=review_note,
            )
            st.success("已采纳教学建议。")
        if reject.button("拒绝此草稿", key=f"teaching_reject_{latest['id']}"):
            service.review_recommendation(
                latest["id"],
                status="rejected",
                reviewed_summary=reviewed_summary,
                review_note=review_note,
            )
            st.warning("已拒绝教学建议草稿。")

    bundle = service.export_case_bundle(dataset["id"], class_code, node_code)
    st.subheader("导出脱敏案例")
    exports = st.columns(3)
    exports[0].download_button(
        "下载案例 JSON",
        data=json.dumps(bundle, ensure_ascii=False, indent=2),
        file_name="班级教学建议案例.json",
        mime="application/json",
        key="teaching_export_json",
    )
    exports[1].download_button(
        "下载案例 Markdown",
        data=recommendation_markdown(bundle),
        file_name="班级教学建议案例.md",
        mime="text/markdown",
        key="teaching_export_markdown",
    )
    for file_name, data in recommendation_csv_sections(bundle).items():
        st.download_button(
            f"下载 {file_name}",
            data=data,
            file_name=file_name,
            mime="text/csv",
            key=f"teaching_export_{file_name}",
        )
