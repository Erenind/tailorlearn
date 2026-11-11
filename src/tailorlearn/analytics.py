from __future__ import annotations

from collections import defaultdict
import json
from typing import Any
import uuid

from tailorlearn.db import Repository, utc_now
from tailorlearn.learning_data import split_multi_value


ANALYTICS_SERVICE_VERSION = "2026-11-11-student-ranking-issues"
ISSUE_SCORE_RATE_THRESHOLD = 0.6


class AnalyticsService:
    """Derive evidence-aware learning metrics from a single dataset."""

    def __init__(self, repository: Repository):
        self.repo = repository

    def _dataset_rows(self, table_name: str, dataset_id: str) -> list[dict[str, Any]]:
        with self.repo.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    f"""
                    SELECT * FROM {table_name}
                    WHERE dataset_id=? AND deleted_at IS NULL
                    ORDER BY row_id
                    """,
                    (dataset_id,),
                ).fetchall()
            ]

    def _latest_answers(self, dataset_id: str) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self._dataset_rows("learning_answers", dataset_id):
            event = str(row["answer_event_code"])
            previous = latest.get(event)
            if previous is None or int(row["revision"] or 0) > int(
                previous["revision"] or 0
            ):
                latest[event] = row
        return list(latest.values())

    @staticmethod
    def _is_valid_scored_answer(
        answer: dict[str, Any],
        *,
        students: set[str],
        materials: set[str],
        nodes: set[str],
        themes: set[str],
    ) -> bool:
        if answer.get("grading_status") != "已评分":
            return False
        score = answer.get("score")
        full_score = answer.get("full_score")
        if score is None or full_score is None:
            return False
        if not 0 <= score <= full_score:
            return False
        return (
            answer.get("student_code") in students
            and answer.get("material_code") in materials
            and answer.get("node_code") in nodes
            and answer.get("theme_code") in themes
        )

    def student_theme_metrics(
        self,
        dataset_id: str,
        student_code: str,
        node_code: str | None = None,
    ) -> list[dict[str, Any]]:
        students = {
            str(row["student_code"])
            for row in self._dataset_rows("learning_students", dataset_id)
        }
        material_rows = self._dataset_rows("learning_materials", dataset_id)
        materials = {str(row["material_code"]) for row in material_rows}
        nodes = {
            str(row["node_code"])
            for row in self._dataset_rows("learning_nodes", dataset_id)
        }
        requirements = self._dataset_rows("learning_requirements", dataset_id)
        theme_names = {
            str(row["theme_code"]): str(row.get("theme_name") or row["theme_code"])
            for row in requirements
            if row.get("theme_code")
        }
        themes = set(theme_names)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for answer in self._latest_answers(dataset_id):
            if answer.get("student_code") != student_code:
                continue
            if node_code is not None and answer.get("node_code") != node_code:
                continue
            grouped[str(answer.get("theme_code") or "")].append(answer)

        metrics: list[dict[str, Any]] = []
        for theme_code, answers in sorted(grouped.items()):
            valid = [
                answer
                for answer in answers
                if self._is_valid_scored_answer(
                    answer,
                    students=students,
                    materials=materials,
                    nodes=nodes,
                    themes=themes,
                )
            ]
            pending = [
                answer
                for answer in answers
                if answer.get("grading_status") == "待评分"
            ]
            unanswered = [
                answer
                for answer in answers
                if answer.get("grading_status") == "未作答"
            ]
            excluded = [
                answer
                for answer in answers
                if answer not in valid
                and answer not in pending
                and answer not in unanswered
            ]
            score_total = sum(int(answer["score"]) for answer in valid)
            full_total = sum(int(answer["full_score"]) for answer in valid)
            scored_records = len(valid)
            if scored_records >= 3:
                evidence_status = "sufficient"
            elif scored_records:
                evidence_status = "limited"
            else:
                evidence_status = "insufficient"
            metrics.append(
                {
                    "theme_code": theme_code,
                    "theme_name": theme_names.get(theme_code, theme_code),
                    "score_rate": score_total / full_total if full_total else None,
                    "scored_records": scored_records,
                    "material_count": len(
                        {answer.get("material_code") for answer in valid}
                    ),
                    "pending_records": len(pending),
                    "unanswered_records": len(unanswered),
                    "excluded_records": len(excluded),
                    "evidence_status": evidence_status,
                }
            )
        return metrics

    def student_performance_ranking(
        self,
        dataset_id: str,
    ) -> list[dict[str, Any]]:
        """Rank students by the arithmetic mean of their observed theme rates."""
        student_rows = self._dataset_rows("learning_students", dataset_id)
        requirement_rows = self._dataset_rows(
            "learning_requirements",
            dataset_id,
        )
        total_theme_count = len(
            {
                str(row["theme_code"])
                for row in requirement_rows
                if row.get("theme_code")
            }
        )
        ranking: list[dict[str, Any]] = []
        for student in student_rows:
            student_code = str(student["student_code"])
            metrics = self.student_theme_metrics(dataset_id, student_code)
            observed = [
                metric
                for metric in metrics
                if metric.get("score_rate") is not None
            ]
            score_rate = (
                sum(float(metric["score_rate"]) for metric in observed)
                / len(observed)
                if observed
                else None
            )
            observed_theme_count = len(observed)
            sufficient_theme_count = sum(
                metric["evidence_status"] == "sufficient"
                for metric in observed
            )
            if score_rate is None:
                evidence_status = "insufficient"
            elif (
                observed_theme_count == total_theme_count
                and sufficient_theme_count == total_theme_count
            ):
                evidence_status = "sufficient"
            else:
                evidence_status = "limited"
            ranking.append(
                {
                    "student_code": student_code,
                    "class_code": str(student.get("class_code") or "未分班"),
                    "grade": student.get("grade"),
                    "score_rate": score_rate,
                    "valid_records": sum(
                        int(metric["scored_records"]) for metric in observed
                    ),
                    "observed_theme_count": observed_theme_count,
                    "total_theme_count": total_theme_count,
                    "evidence_status": evidence_status,
                    "rank": None,
                }
            )

        ranked_rates = sorted(
            {
                round(float(row["score_rate"]), 12)
                for row in ranking
                if row["score_rate"] is not None
            },
            reverse=True,
        )
        rank_by_rate = {
            score_rate: index + 1
            for index, score_rate in enumerate(ranked_rates)
        }
        for row in ranking:
            if row["score_rate"] is not None:
                row["rank"] = rank_by_rate[round(float(row["score_rate"]), 12)]
        return sorted(
            ranking,
            key=lambda row: (
                row["score_rate"] is None,
                -float(row["score_rate"] or 0),
                str(row["student_code"]),
            ),
        )

    def _theme_issue_evidence(
        self,
        dataset_id: str,
        *,
        theme_code: str,
        student_code: str,
        node_code: str | None,
        class_code: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        requirement_rows = [
            row
            for row in self._dataset_rows("learning_requirements", dataset_id)
            if row.get("theme_code") == theme_code
        ]
        source_rows = [
            {
                "主题": row.get("theme_name") or theme_code,
                "观察指标": row.get("indicator_name") or row["indicator_code"],
                "指标编号": row["indicator_code"],
                "主题目标": row.get("theme_goal") or "",
                "课标依据": " · ".join(
                    value
                    for value in (
                        str(row.get("standard_ref") or ""),
                        (
                            f"第 {row['standard_page']} 页"
                            if row.get("standard_page")
                            else ""
                        ),
                    )
                    if value
                ),
                "教材正文": (
                    f"第 {row['textbook_page']} 页"
                    if row.get("textbook_page")
                    else ""
                ),
            }
            for row in requirement_rows
        ]
        student_rows = self._dataset_rows("learning_students", dataset_id)
        student_by_code = {
            str(row["student_code"]): row for row in student_rows
        }
        material_by_code = {
            str(row["material_code"]): row
            for row in self._dataset_rows("learning_materials", dataset_id)
        }
        nodes = {
            str(row["node_code"])
            for row in self._dataset_rows("learning_nodes", dataset_id)
        }
        themes = {
            str(row["theme_code"])
            for row in self._dataset_rows("learning_requirements", dataset_id)
            if row.get("theme_code")
        }
        records: list[dict[str, Any]] = []
        for answer in self._latest_answers(dataset_id):
            answer_student = str(answer.get("student_code") or "")
            if class_code is not None:
                student = student_by_code.get(answer_student)
                if (
                    student is None
                    or str(student.get("class_code") or "未分班") != class_code
                ):
                    continue
            elif answer_student != student_code:
                continue
            if answer.get("theme_code") != theme_code:
                continue
            if node_code is not None and answer.get("node_code") != node_code:
                continue
            if not self._is_valid_scored_answer(
                answer,
                students=set(student_by_code),
                materials=set(material_by_code),
                nodes=nodes,
                themes=themes,
            ):
                continue
            material = material_by_code[str(answer["material_code"])]
            records.append(
                {
                    "学生编号": answer_student,
                    "记录编号": answer["record_code"],
                    "节点编号": answer.get("node_code") or "",
                    "材料编号": answer["material_code"],
                    "题干或内容": material.get("content") or "",
                    "得分": answer["score"],
                    "满分": answer["full_score"],
                    "题目来源": material.get("source") or "",
                    "提交时间": answer.get("submitted_at") or "",
                }
            )
        return {"sources": source_rows, "records": records}

    def student_issue_analysis(
        self,
        dataset_id: str,
        student_code: str,
        node_code: str | None = None,
    ) -> dict[str, Any]:
        """Classify evidence-backed student and class issues deterministically."""
        student_rows = self._dataset_rows("learning_students", dataset_id)
        student = next(
            (
                row
                for row in student_rows
                if str(row["student_code"]) == student_code
            ),
            None,
        )
        if student is None:
            raise ValueError("找不到当前学生")
        class_code = str(student.get("class_code") or "未分班")
        student_metrics = {
            str(metric["theme_code"]): metric
            for metric in self.student_theme_metrics(
                dataset_id,
                student_code,
                node_code,
            )
        }
        class_metrics = {
            str(metric["theme_code"]): metric
            for metric in self.class_theme_metrics(dataset_id, node_code)
            if metric["class_code"] == class_code
        }

        def evidence(theme_code: str, *, common: bool = False) -> dict[str, Any]:
            return self._theme_issue_evidence(
                dataset_id,
                theme_code=theme_code,
                student_code=student_code,
                node_code=node_code,
                class_code=class_code if common else None,
            )

        need_consolidation: list[dict[str, Any]] = []
        personalized: list[dict[str, Any]] = []
        for theme_code, metric in sorted(student_metrics.items()):
            score_rate = metric.get("score_rate")
            if (
                metric["evidence_status"] != "sufficient"
                or score_rate is None
            ):
                continue
            if float(score_rate) < ISSUE_SCORE_RATE_THRESHOLD:
                need_consolidation.append(
                    {
                        "theme_code": theme_code,
                        "theme_name": metric["theme_name"],
                        "score_rate": float(score_rate),
                        "reason": (
                            f"当前主题得分率为 {float(score_rate):.1%}，"
                            f"低于 {ISSUE_SCORE_RATE_THRESHOLD:.0%} 的巩固阈值。"
                        ),
                        **evidence(theme_code),
                    }
                )
            class_metric = class_metrics.get(theme_code)
            if (
                class_metric is None
                or class_metric.get("score_rate") is None
                or class_metric["evidence_status"] != "sufficient"
            ):
                continue
            class_rate = float(class_metric["score_rate"])
            if float(score_rate) < class_rate:
                personalized.append(
                    {
                        "theme_code": theme_code,
                        "theme_name": metric["theme_name"],
                        "score_rate": float(score_rate),
                        "class_score_rate": class_rate,
                        "delta": float(score_rate) - class_rate,
                        "reason": (
                            f"低于 {class_code} 同节点平均 "
                            f"{float(score_rate) - class_rate:+.1%}。"
                        ),
                        **evidence(theme_code),
                    }
                )

        common: list[dict[str, Any]] = []
        for theme_code, metric in sorted(class_metrics.items()):
            score_rate = metric.get("score_rate")
            if (
                metric["evidence_status"] != "sufficient"
                or score_rate is None
                or float(score_rate) >= ISSUE_SCORE_RATE_THRESHOLD
            ):
                continue
            common.append(
                {
                    "theme_code": theme_code,
                    "theme_name": metric["theme_name"],
                    "class_code": class_code,
                    "score_rate": float(score_rate),
                    "student_count": int(metric["student_count"]),
                    "scored_records": int(metric["scored_records"]),
                    "reason": (
                        f"{class_code} 同节点平均得分率为 "
                        f"{float(score_rate):.1%}，低于 "
                        f"{ISSUE_SCORE_RATE_THRESHOLD:.0%}。"
                    ),
                    **evidence(theme_code, common=True),
                }
            )

        supplemental = [
            {
                "theme_code": theme_code,
                "theme_name": metric["theme_name"],
                "evidence_status": metric["evidence_status"],
                "scored_records": int(metric["scored_records"]),
                "pending_records": int(metric.get("pending_records", 0)),
                "unanswered_records": int(metric.get("unanswered_records", 0)),
                "reason": (
                    f"仅有 {int(metric['scored_records'])} 条有效评分记录，"
                    "未达到主题巩固判定条件。"
                ),
            }
            for theme_code, metric in sorted(student_metrics.items())
            if metric["evidence_status"] != "sufficient"
        ]
        return {
            "class_code": class_code,
            "threshold": ISSUE_SCORE_RATE_THRESHOLD,
            "need_consolidation": need_consolidation,
            "personalized": personalized,
            "common": common,
            "supplemental": supplemental,
        }

    def student_theme_trend(
        self,
        dataset_id: str,
        student_code: str,
        theme_code: str,
    ) -> dict[str, Any]:
        """Return chronological observed performance without inferring mastery."""
        nodes = sorted(
            self._dataset_rows("learning_nodes", dataset_id),
            key=lambda row: (str(row.get("test_date") or ""), str(row["node_code"])),
        )
        points: list[dict[str, Any]] = []
        for node in nodes:
            metric = next(
                (
                    item
                    for item in self.student_theme_metrics(
                        dataset_id,
                        student_code,
                        str(node["node_code"]),
                    )
                    if item["theme_code"] == theme_code
                ),
                None,
            )
            if metric is None:
                metric = {
                    "theme_code": theme_code,
                    "theme_name": theme_code,
                    "score_rate": None,
                    "scored_records": 0,
                    "material_count": 0,
                    "pending_records": 0,
                    "unanswered_records": 0,
                    "excluded_records": 0,
                    "evidence_status": "insufficient",
                }
            points.append(
                {
                    "node_code": node["node_code"],
                    "test_date": node.get("test_date"),
                    **metric,
                }
            )
        if len(points) < 2:
            return {"status": "insufficient", "delta": None, "points": points}
        predecessor, successor = points[-2:]
        if (
            predecessor["evidence_status"] != "sufficient"
            or successor["evidence_status"] != "sufficient"
        ):
            return {"status": "insufficient", "delta": None, "points": points}
        return {
            "status": "observable",
            "delta": successor["score_rate"] - predecessor["score_rate"],
            "points": points,
        }

    def class_theme_metrics(
        self,
        dataset_id: str,
        node_code: str | None = None,
    ) -> list[dict[str, Any]]:
        student_rows = self._dataset_rows("learning_students", dataset_id)
        students = {str(row["student_code"]) for row in student_rows}
        classes = {
            str(row["student_code"]): str(row.get("class_code") or "未分班")
            for row in student_rows
        }
        materials = {
            str(row["material_code"])
            for row in self._dataset_rows("learning_materials", dataset_id)
        }
        nodes = {
            str(row["node_code"])
            for row in self._dataset_rows("learning_nodes", dataset_id)
        }
        requirements = self._dataset_rows("learning_requirements", dataset_id)
        theme_names = {
            str(row["theme_code"]): str(row.get("theme_name") or row["theme_code"])
            for row in requirements
            if row.get("theme_code")
        }
        themes = set(theme_names)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for answer in self._latest_answers(dataset_id):
            if node_code is not None and answer.get("node_code") != node_code:
                continue
            student_code = str(answer.get("student_code") or "")
            if student_code in classes:
                grouped[(classes[student_code], str(answer.get("theme_code") or ""))].append(
                    answer
                )

        metrics: list[dict[str, Any]] = []
        for (class_code, theme_code), answers in sorted(grouped.items()):
            valid = [
                answer
                for answer in answers
                if self._is_valid_scored_answer(
                    answer,
                    students=students,
                    materials=materials,
                    nodes=nodes,
                    themes=themes,
                )
            ]
            pending = [
                answer
                for answer in answers
                if answer.get("grading_status") == "待评分"
            ]
            unanswered = [
                answer
                for answer in answers
                if answer.get("grading_status") == "未作答"
            ]
            excluded = [
                answer
                for answer in answers
                if answer not in valid
                and answer not in pending
                and answer not in unanswered
            ]
            score_total = sum(int(answer["score"]) for answer in valid)
            full_total = sum(int(answer["full_score"]) for answer in valid)
            student_count = len({answer.get("student_code") for answer in valid})
            if student_count >= 5 and len(valid) >= 10:
                evidence_status = "sufficient"
            elif valid:
                evidence_status = "limited"
            else:
                evidence_status = "insufficient"
            metrics.append(
                {
                    "class_code": class_code,
                    "theme_code": theme_code,
                    "theme_name": theme_names.get(theme_code, theme_code),
                    "score_rate": score_total / full_total if full_total else None,
                    "student_count": student_count,
                    "scored_records": len(valid),
                    "pending_records": len(pending),
                    "unanswered_records": len(unanswered),
                    "excluded_records": len(excluded),
                    "evidence_status": evidence_status,
                }
            )
        return metrics

    def class_theme_distribution(
        self,
        dataset_id: str,
        node_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return only student theme rates backed by the minimum evidence threshold."""
        classes = {
            str(row["student_code"]): str(row.get("class_code") or "未分班")
            for row in self._dataset_rows("learning_students", dataset_id)
        }
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for student_code, class_code in sorted(classes.items()):
            for metric in self.student_theme_metrics(dataset_id, student_code, node_code):
                if (
                    metric["evidence_status"] != "sufficient"
                    or metric["score_rate"] is None
                ):
                    continue
                grouped[
                    (
                        class_code,
                        str(metric["theme_code"]),
                        str(metric["theme_name"]),
                    )
                ].append(float(metric["score_rate"]))
        return [
            {
                "class_code": class_code,
                "theme_code": theme_code,
                "theme_name": theme_name,
                "observed_students": len(rates),
                "student_rates": rates,
            }
            for (class_code, theme_code, theme_name), rates in sorted(grouped.items())
        ]

    def supplemental_observations(
        self,
        dataset_id: str,
        student_code: str,
        node_code: str,
    ) -> list[dict[str, Any]]:
        """Suggest only supplied materials when evidence for an indicator is thin."""
        students = {
            str(row["student_code"])
            for row in self._dataset_rows("learning_students", dataset_id)
        }
        material_rows = self._dataset_rows("learning_materials", dataset_id)
        material_by_code = {
            str(row["material_code"]): row for row in material_rows
        }
        nodes = {
            str(row["node_code"])
            for row in self._dataset_rows("learning_nodes", dataset_id)
        }
        requirements = self._dataset_rows("learning_requirements", dataset_id)
        themes = {
            str(row["theme_code"])
            for row in requirements
            if row.get("theme_code")
        }
        valid_answers = [
            answer
            for answer in self._latest_answers(dataset_id)
            if answer.get("student_code") == student_code
            and answer.get("node_code") == node_code
            and self._is_valid_scored_answer(
                answer,
                students=students,
                materials=set(material_by_code),
                nodes=nodes,
                themes=themes,
            )
        ]
        answered_materials = {
            str(answer["material_code"]) for answer in valid_answers
        }
        node = next(
            (
                row
                for row in self._dataset_rows("learning_nodes", dataset_id)
                if row["node_code"] == node_code
            ),
            None,
        )
        available_materials = set(
            split_multi_value(node.get("available_materials")) if node else []
        )
        suggestions: list[dict[str, Any]] = []
        for requirement in sorted(
            requirements,
            key=lambda row: (str(row.get("theme_code") or ""), str(row["indicator_code"])),
        ):
            indicator_code = str(requirement["indicator_code"])
            evidence_count = sum(
                indicator_code
                in split_multi_value(
                    material_by_code[str(answer["material_code"])].get(
                        "indicator_codes"
                    )
                )
                for answer in valid_answers
                if str(answer["material_code"]) in material_by_code
            )
            if evidence_count >= 3:
                continue
            candidates = [
                code
                for code, material in sorted(material_by_code.items())
                if code not in answered_materials
                and code in available_materials
                and material.get("material_type") in {"题目", "学习卡"}
                and indicator_code in split_multi_value(material.get("indicator_codes"))
            ]
            if candidates:
                suggestions.append(
                    {
                        "theme_code": requirement.get("theme_code"),
                        "indicator_code": indicator_code,
                        "reason": f"有效记录不足 3 条，当前仅有 {evidence_count} 条。",
                        "candidate_material_codes": candidates,
                    }
                )
        return suggestions

    def _metrics_for_scope(
        self,
        dataset_id: str,
        scope: dict[str, Any],
    ) -> list[dict[str, Any]]:
        student_code = scope.get("student_code")
        node_code = scope.get("node_code")
        if student_code:
            return self.student_theme_metrics(dataset_id, str(student_code), node_code)
        class_code = scope.get("class_code")
        metrics = self.class_theme_metrics(dataset_id, node_code)
        if class_code:
            metrics = [
                metric for metric in metrics if metric["class_code"] == class_code
            ]
        return metrics

    def create_snapshot(
        self,
        dataset_id: str,
        scope: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = self._metrics_for_scope(dataset_id, scope)
        snapshot = {
            "id": f"as_{uuid.uuid4().hex[:12]}",
            "dataset_id": dataset_id,
            "scope": scope,
            "metrics": metrics,
        }
        with self.repo.connect() as connection:
            cutoff = connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM learning_change_log WHERE dataset_id=?",
                (dataset_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO analysis_snapshots
                (id, dataset_id, scope_json, metrics_json, change_log_cutoff, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot["id"],
                    dataset_id,
                    json.dumps(scope, ensure_ascii=False, sort_keys=True),
                    json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                    cutoff,
                    utc_now(),
                ),
            )
        return snapshot

    def list_snapshots(self, dataset_id: str) -> list[dict[str, Any]]:
        with self.repo.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, scope_json FROM analysis_snapshots
                WHERE dataset_id=?
                ORDER BY created_at DESC, id DESC
                """,
                (dataset_id,),
            ).fetchall()
        return [
            {"id": row["id"], "scope": json.loads(row["scope_json"])}
            for row in rows
        ]

    def compare_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self.repo.connect() as connection:
            row = connection.execute(
                "SELECT * FROM analysis_snapshots WHERE id=?", (snapshot_id,)
            ).fetchone()
            if row is None:
                raise ValueError("找不到分析快照")
            snapshot = dict(row)
            change_rows = connection.execute(
                """
                SELECT after_json FROM learning_change_log
                WHERE dataset_id=? AND id>? AND data_type='answer'
                ORDER BY id
                """,
                (snapshot["dataset_id"], snapshot["change_log_cutoff"]),
            ).fetchall()
        scope = json.loads(snapshot["scope_json"])
        before_metrics = json.loads(snapshot["metrics_json"])
        after_metrics = self._metrics_for_scope(snapshot["dataset_id"], scope)
        before_by_theme = {
            str(metric["theme_code"]): metric for metric in before_metrics
        }
        after_by_theme = {
            str(metric["theme_code"]): metric for metric in after_metrics
        }
        changes: list[dict[str, Any]] = []
        for theme_code in sorted(set(before_by_theme) | set(after_by_theme)):
            before = before_by_theme.get(theme_code, {})
            after = after_by_theme.get(theme_code, {})
            if (
                before.get("score_rate") != after.get("score_rate")
                or before.get("evidence_status") != after.get("evidence_status")
            ):
                changes.append(
                    {
                        "theme_code": theme_code,
                        "before_score_rate": before.get("score_rate"),
                        "after_score_rate": after.get("score_rate"),
                        "before_evidence_status": before.get("evidence_status"),
                        "after_evidence_status": after.get("evidence_status"),
                    }
                )
        affected_record_codes = [
            str(after["record_code"])
            for row in change_rows
            if (after := json.loads(row["after_json"] or "{}"))
            and after.get("record_code")
        ]
        return {
            "snapshot_id": snapshot_id,
            "changes": changes,
            "affected_record_codes": affected_record_codes,
        }

    def theme_evidence(
        self,
        dataset_id: str,
        student_code: str,
        node_code: str,
        theme_code: str,
    ) -> dict[str, Any]:
        material_rows = self._dataset_rows("learning_materials", dataset_id)
        material_by_code = {
            str(row["material_code"]): row for row in material_rows
        }
        requirements = [
            row
            for row in self._dataset_rows("learning_requirements", dataset_id)
            if row.get("theme_code") == theme_code
        ]
        students = {
            str(row["student_code"])
            for row in self._dataset_rows("learning_students", dataset_id)
        }
        nodes = {
            str(row["node_code"])
            for row in self._dataset_rows("learning_nodes", dataset_id)
        }
        themes = {
            str(row.get("theme_code"))
            for row in self._dataset_rows("learning_requirements", dataset_id)
            if row.get("theme_code")
        }
        records: list[dict[str, Any]] = []
        for answer in self._latest_answers(dataset_id):
            if (
                answer.get("student_code") != student_code
                or answer.get("node_code") != node_code
                or answer.get("theme_code") != theme_code
                or not self._is_valid_scored_answer(
                    answer,
                    students=students,
                    materials=set(material_by_code),
                    nodes=nodes,
                    themes=themes,
                )
            ):
                continue
            material = material_by_code[str(answer["material_code"])]
            records.append(
                {
                    "record_code": answer["record_code"],
                    "material_code": answer["material_code"],
                    "score": answer["score"],
                    "full_score": answer["full_score"],
                    "indicator_codes": split_multi_value(
                        material.get("indicator_codes")
                    ),
                }
            )
        return {
            "requirement_ids": [
                str(row["indicator_code"]) for row in requirements
            ],
            "records": records,
        }

    def student_indicator_metrics(
        self,
        dataset_id: str,
        student_code: str,
        node_code: str | None = None,
    ) -> list[dict[str, Any]]:
        requirements = self._dataset_rows("learning_requirements", dataset_id)
        material_by_code = {
            str(row["material_code"]): row
            for row in self._dataset_rows("learning_materials", dataset_id)
        }
        students = {
            str(row["student_code"])
            for row in self._dataset_rows("learning_students", dataset_id)
        }
        nodes = {
            str(row["node_code"])
            for row in self._dataset_rows("learning_nodes", dataset_id)
        }
        themes = {
            str(row["theme_code"])
            for row in requirements
            if row.get("theme_code")
        }
        valid_answers = [
            answer
            for answer in self._latest_answers(dataset_id)
            if answer.get("student_code") == student_code
            and (node_code is None or answer.get("node_code") == node_code)
            and self._is_valid_scored_answer(
                answer,
                students=students,
                materials=set(material_by_code),
                nodes=nodes,
                themes=themes,
            )
        ]
        metrics: list[dict[str, Any]] = []
        for requirement in sorted(requirements, key=lambda row: row["indicator_code"]):
            indicator_code = str(requirement["indicator_code"])
            matching = [
                answer
                for answer in valid_answers
                if indicator_code
                in split_multi_value(
                    material_by_code[str(answer["material_code"])].get(
                        "indicator_codes"
                    )
                )
            ]
            score_total = sum(int(answer["score"]) for answer in matching)
            full_total = sum(int(answer["full_score"]) for answer in matching)
            scored_records = len(matching)
            metrics.append(
                {
                    "indicator_code": indicator_code,
                    "indicator_name": requirement.get("indicator_name")
                    or indicator_code,
                    "theme_code": requirement.get("theme_code"),
                    "score_rate": score_total / full_total if full_total else None,
                    "scored_records": scored_records,
                    "material_count": len(
                        {answer["material_code"] for answer in matching}
                    ),
                    "evidence_status": (
                        "sufficient"
                        if scored_records >= 3
                        else "limited"
                        if scored_records
                        else "insufficient"
                    ),
                }
            )
        return metrics

    def typical_case_bundle(
        self,
        dataset_id: str,
        student_code: str,
        node_code: str | None = None,
    ) -> dict[str, Any]:
        """Build an exportable, evidence-linked case package for one student."""
        themes = self.student_theme_metrics(dataset_id, student_code, node_code)
        return {
            "dataset_id": dataset_id,
            "student_code": student_code,
            "node_code": node_code,
            "themes": themes,
            "indicators": self.student_indicator_metrics(
                dataset_id, student_code, node_code
            ),
            "trends": {
                str(metric["theme_code"]): self.student_theme_trend(
                    dataset_id, student_code, str(metric["theme_code"])
                )
                for metric in themes
            },
            "evidence": (
                {
                    str(metric["theme_code"]): self.theme_evidence(
                        dataset_id,
                        student_code,
                        node_code,
                        str(metric["theme_code"]),
                    )
                    for metric in themes
                }
                if node_code is not None
                else {}
            ),
        }
