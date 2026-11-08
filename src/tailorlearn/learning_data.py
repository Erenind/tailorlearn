from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from tailorlearn.db import Repository, utc_now


EXTRA_FIELD = "__extra__"
IGNORE_FIELD = "__ignore__"


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    value_type: str = "text"
    required: bool = False
    multiline: bool = False


@dataclass(frozen=True)
class DataTypeSpec:
    key: str
    label: str
    table_name: str
    source_table_name: str
    fields: tuple[FieldSpec, ...]
    natural_key: tuple[str, ...]
    extra_unique_key: tuple[str, ...] = ()

    @property
    def field_by_key(self) -> dict[str, FieldSpec]:
        return {item.key: item for item in self.fields}

    @property
    def label_by_key(self) -> dict[str, str]:
        return {item.key: item.label for item in self.fields}


STUDENT_SPEC = DataTypeSpec(
    key="student",
    label="模拟学生",
    table_name="learning_students",
    source_table_name="模拟学生",
    fields=(
        FieldSpec("student_code", "学生编号", required=True),
        FieldSpec("class_code", "模拟班级编号", required=True),
        FieldSpec("grade", "年级", "integer", required=True),
    ),
    natural_key=("student_code",),
)

REQUIREMENT_SPEC = DataTypeSpec(
    key="requirement",
    label="课程要求",
    table_name="learning_requirements",
    source_table_name="课程要求",
    fields=(
        FieldSpec("theme_code", "主题编号", required=True),
        FieldSpec("theme_name", "主题", required=True),
        FieldSpec("theme_goal", "主题目标", required=True, multiline=True),
        FieldSpec("indicator_code", "指标编号", required=True),
        FieldSpec("indicator_name", "观察指标", required=True, multiline=True),
        FieldSpec("standard_ref", "课标整理编号", required=True),
        FieldSpec("standard_page", "课标文件页／正文页", required=True),
        FieldSpec("textbook_page", "教材正文页", required=True),
    ),
    natural_key=("indicator_code",),
)

MATERIAL_SPEC = DataTypeSpec(
    key="material",
    label="题目与材料",
    table_name="learning_materials",
    source_table_name="题目与材料",
    fields=(
        FieldSpec("material_code", "材料编号", required=True),
        FieldSpec("theme_code", "主题编号"),
        FieldSpec("material_type", "材料类型", required=True),
        FieldSpec("question_type", "题型"),
        FieldSpec("content", "题干或内容", required=True, multiline=True),
        FieldSpec("options", "选项", multiline=True),
        FieldSpec("reference_answer", "参考答案", multiline=True),
        FieldSpec("reference_solution", "参考解析", multiline=True),
        FieldSpec("full_score", "满分", "integer"),
        FieldSpec("rubric", "评分点", multiline=True),
        FieldSpec("indicator_codes", "指标编号"),
        FieldSpec("source", "来源或资料文件", required=True),
    ),
    natural_key=("material_code",),
)

NODE_SPEC = DataTypeSpec(
    key="node",
    label="测验节点",
    table_name="learning_nodes",
    source_table_name="测验节点",
    fields=(
        FieldSpec("node_code", "节点编号", required=True),
        FieldSpec("test_date", "模拟测验日期", "date", required=True),
        FieldSpec("taught_topics", "已教及测量主题", required=True),
        FieldSpec("description", "说明", required=True, multiline=True),
        FieldSpec("available_materials", "可用题目编号", required=True),
    ),
    natural_key=("node_code",),
)

ANSWER_SPEC = DataTypeSpec(
    key="answer",
    label="作答记录",
    table_name="learning_answers",
    source_table_name="作答记录",
    fields=(
        FieldSpec("record_code", "记录编号", required=True),
        FieldSpec("student_code", "学生编号", required=True),
        FieldSpec("material_code", "材料编号", required=True),
        FieldSpec("theme_code", "主题编号", required=True),
        FieldSpec("node_code", "节点编号", required=True),
        FieldSpec("answer_event_code", "作答事件编号", required=True),
        FieldSpec("revision", "修订版本", "integer", required=True),
        FieldSpec(
            "submitted_at",
            "模拟提交时间",
            "datetime",
            required=True,
        ),
        FieldSpec("grading_status", "评分状态", required=True),
        FieldSpec("score", "得分", "integer"),
        FieldSpec("full_score", "满分", "integer", required=True),
        FieldSpec("final_answer", "学生最终作答", multiline=True),
        FieldSpec("solution", "解答过程", multiline=True),
        FieldSpec("rubric_scores", "评分点得分", multiline=True),
    ),
    natural_key=("record_code",),
    extra_unique_key=("answer_event_code", "revision"),
)

FIELD_SPEC = DataTypeSpec(
    key="field_spec",
    label="字段说明",
    table_name="learning_field_definitions",
    source_table_name="字段说明",
    fields=(
        FieldSpec("table_name", "数据表", required=True),
        FieldSpec("field_name", "字段名", required=True),
        FieldSpec("field_type", "类型", required=True),
        FieldSpec("meaning", "含义", required=True, multiline=True),
        FieldSpec("allowed_format", "取值或格式", multiline=True),
        FieldSpec("example", "示例", required=True, multiline=True),
        FieldSpec("blank_behavior", "为空时", required=True),
    ),
    natural_key=("table_name", "field_name"),
)

DATA_TYPE_SPECS = {
    spec.key: spec
    for spec in (
        STUDENT_SPEC,
        REQUIREMENT_SPEC,
        MATERIAL_SPEC,
        NODE_SPEC,
        ANSWER_SPEC,
        FIELD_SPEC,
    )
}
DATA_TYPES_BY_LABEL = {spec.label: spec for spec in DATA_TYPE_SPECS.values()}
GRADING_STATUSES = {"已评分", "待评分", "未作答"}
TYPE_LABELS = {
    "text": "文本",
    "integer": "整数",
    "date": "日期",
    "datetime": "日期时间",
}


@dataclass
class CSVTable:
    headers: list[str]
    rows: list[tuple[int, dict[str, str]]]
    encoding: str


@dataclass
class ImportPreview:
    dataset_id: str
    data_type: str
    file_name: str
    content: bytes
    headers: list[str]
    mapping: dict[str, str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    extras: list[dict[str, str]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def spec(self) -> DataTypeSpec:
        return DATA_TYPE_SPECS[self.data_type]


@dataclass
class ImportResult:
    batch_id: str
    inserted: int
    updated: int
    skipped: int
    conflicts: int
    issues: int


def normalize_header(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def split_multi_value(value: Any) -> list[str]:
    if value is None:
        return []
    return [
        token.strip()
        for token in re.split(r"[、,，;；]", str(value))
        if token.strip()
    ]


def new_dataset_id() -> str:
    return f"lds_{uuid.uuid4().hex[:12]}"


def new_batch_id() -> str:
    return f"ldb_{uuid.uuid4().hex[:12]}"


class LearningDataService:
    def __init__(self, repository: Repository, db_dir: Path):
        self.repo = repository
        self.db_dir = Path(db_dir)

    def list_datasets(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        where = "" if include_deleted else "WHERE deleted_at IS NULL"
        with self.repo.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    f"""
                    SELECT * FROM learning_datasets
                    {where}
                    ORDER BY created_at DESC, name
                    """
                ).fetchall()
            ]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self.repo.connect() as connection:
            row = connection.execute(
                "SELECT * FROM learning_datasets WHERE id=?",
                (dataset_id,),
            ).fetchone()
            return dict(row) if row else None

    def create_dataset(self, name: str, description: str = "") -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("学情数据集名称不能为空")
        dataset_id = new_dataset_id()
        now = utc_now()
        with self.repo.connect() as connection:
            connection.execute(
                """
                INSERT INTO learning_datasets
                (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (dataset_id, name, description.strip() or None, now, now),
            )
        return self.get_dataset(dataset_id) or {}

    def update_dataset(
        self,
        dataset_id: str,
        *,
        name: str,
        description: str = "",
    ) -> None:
        name = name.strip()
        if not name:
            raise ValueError("学情数据集名称不能为空")
        with self.repo.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE learning_datasets
                SET name=?, description=?, updated_at=?
                WHERE id=? AND deleted_at IS NULL
                """,
                (
                    name,
                    description.strip() or None,
                    utc_now(),
                    dataset_id,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError("找不到该学情数据集")

    def read_csv(self, content: bytes) -> CSVTable:
        if not content:
            raise ValueError("CSV 文件为空")
        text = ""
        encoding = ""
        for candidate in ("utf-8-sig", "gb18030"):
            try:
                text = content.decode(candidate)
                encoding = candidate
                break
            except UnicodeDecodeError:
                continue
        if not encoding:
            raise ValueError("无法识别 CSV 编码，请使用 UTF-8 或 GB18030")
        reader = csv.DictReader(io.StringIO(text))
        headers = list(reader.fieldnames or [])
        if not headers or not any(header.strip() for header in headers):
            raise ValueError("CSV 首行必须是字段名")
        normalized = [normalize_header(header) for header in headers]
        if len(normalized) != len(set(normalized)):
            raise ValueError("CSV 存在重复字段名")
        rows: list[tuple[int, dict[str, str]]] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"第 {row_number} 行包含超出表头数量的单元格")
            values = {
                key: (value if value is not None else "")
                for key, value in row.items()
            }
            if not any(value.strip() for value in values.values()):
                continue
            rows.append((row_number, values))
        if not rows:
            raise ValueError("CSV 没有数据行")
        return CSVTable(headers=headers, rows=rows, encoding=encoding)

    def load_aliases(
        self,
        dataset_id: str,
        data_type: str,
    ) -> dict[str, str]:
        with self.repo.connect() as connection:
            rows = connection.execute(
                """
                SELECT source_name, target_field
                FROM learning_field_aliases
                WHERE dataset_id=? AND data_type=?
                """,
                (dataset_id, data_type),
            ).fetchall()
        return {row["source_name"]: row["target_field"] for row in rows}

    def suggest_mapping(
        self,
        dataset_id: str,
        data_type: str,
        headers: Sequence[str],
    ) -> dict[str, str]:
        spec = DATA_TYPE_SPECS[data_type]
        labels = {
            normalize_header(item.label): item.key for item in spec.fields
        }
        aliases = self.load_aliases(dataset_id, data_type)
        result: dict[str, str] = {}
        for header in headers:
            normalized = normalize_header(header)
            target = labels.get(normalized) or aliases.get(normalized)
            result[header] = target if target in spec.field_by_key else EXTRA_FIELD
        return result

    def mapping_errors(
        self,
        data_type: str,
        mapping: dict[str, str],
    ) -> list[str]:
        spec = DATA_TYPE_SPECS[data_type]
        errors: list[str] = []
        targets = [
            target
            for target in mapping.values()
            if target not in {EXTRA_FIELD, IGNORE_FIELD}
        ]
        duplicates = sorted(
            target for target, count in Counter(targets).items() if count > 1
        )
        if duplicates:
            labels = spec.label_by_key
            errors.append(
                "多个源字段映射到了同一目标字段："
                + "、".join(labels.get(item, item) for item in duplicates)
            )
        invalid_targets = sorted(
            target
            for target in targets
            if target not in spec.field_by_key
        )
        if invalid_targets:
            errors.append("存在无效目标字段：" + "、".join(invalid_targets))
        missing = [
            item.label
            for item in spec.fields
            if item.key not in targets
        ]
        if missing:
            errors.append("缺少必需字段：" + "、".join(missing))
        return errors

    def _coerce(self, value: Any, field_spec: FieldSpec) -> Any:
        if value is None:
            return None
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        text = str(value).strip()
        if text == "":
            return None
        if field_spec.value_type == "integer":
            try:
                return int(text)
            except ValueError as exc:
                raise ValueError(
                    f"{field_spec.label}必须是整数，当前为 {text!r}"
                ) from exc
        if field_spec.value_type == "date":
            try:
                return date.fromisoformat(text).isoformat()
            except ValueError as exc:
                raise ValueError(
                    f"{field_spec.label}必须是 YYYY-MM-DD 日期，当前为 {text!r}"
                ) from exc
        if field_spec.value_type == "datetime":
            try:
                return datetime.fromisoformat(text).isoformat()
            except ValueError as exc:
                raise ValueError(
                    f"{field_spec.label}必须是 ISO 8601 日期时间，"
                    f"当前为 {text!r}"
                ) from exc
        return text

    def _business_issues(
        self,
        data_type: str,
        values: dict[str, Any],
        row_number: int,
    ) -> list[dict[str, Any]]:
        spec = DATA_TYPE_SPECS[data_type]
        issues: list[dict[str, Any]] = []
        for item in spec.fields:
            if item.required and values.get(item.key) in (None, ""):
                issues.append(
                    {
                        "row_number": row_number,
                        "severity": "warning",
                        "code": "blank_required",
                        "field_name": item.key,
                        "message": f"{item.label}为空",
                        "raw_value": "",
                    }
                )
        if data_type == "answer":
            status = values.get("grading_status")
            if status not in GRADING_STATUSES:
                issues.append(
                    {
                        "row_number": row_number,
                        "severity": "warning",
                        "code": "unknown_grading_status",
                        "field_name": "grading_status",
                        "message": f"未知评分状态：{status}",
                        "raw_value": status,
                    }
                )
            score = values.get("score")
            full_score = values.get("full_score")
            if (
                score is not None
                and full_score is not None
                and not 0 <= score <= full_score
            ):
                issues.append(
                    {
                        "row_number": row_number,
                        "severity": "warning",
                        "code": "score_out_of_range",
                        "field_name": "score",
                        "message": f"得分 {score} 不在 0—{full_score} 范围内",
                        "raw_value": score,
                    }
                )
            if status in {"待评分", "未作答"} and score is not None:
                issues.append(
                    {
                        "row_number": row_number,
                        "severity": "warning",
                        "code": "score_status_mismatch",
                        "field_name": "score",
                        "message": f"{status}记录的得分应为空",
                        "raw_value": score,
                    }
                )
            if status == "已评分" and score is None:
                issues.append(
                    {
                        "row_number": row_number,
                        "severity": "warning",
                        "code": "score_status_mismatch",
                        "field_name": "score",
                        "message": "已评分记录的得分不能为空",
                        "raw_value": "",
                    }
                )
        if data_type == "field_spec":
            table_name = values.get("table_name")
            if table_name not in DATA_TYPES_BY_LABEL:
                issues.append(
                    {
                        "row_number": row_number,
                        "severity": "warning",
                        "code": "unknown_data_table",
                        "field_name": "table_name",
                        "message": f"字段说明引用了未知数据表：{table_name}",
                        "raw_value": table_name,
                    }
                )
        return issues

    def _existing_rows(
        self,
        dataset_id: str,
        data_type: str,
    ) -> dict[tuple[Any, ...], dict[str, Any]]:
        spec = DATA_TYPE_SPECS[data_type]
        rows = self.list_records(
            dataset_id,
            data_type,
            include_deleted=True,
        )
        return {
            tuple(row.get(key) for key in spec.natural_key): row
            for row in rows
        }

    def _event_conflict(
        self,
        dataset_id: str,
        values: dict[str, Any],
    ) -> dict[str, Any] | None:
        spec = ANSWER_SPEC
        if not spec.extra_unique_key:
            return None
        with self.repo.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM learning_answers
                WHERE dataset_id=? AND answer_event_code=? AND revision=?
                """,
                (
                    dataset_id,
                    values.get("answer_event_code"),
                    values.get("revision"),
                ),
            ).fetchone()
        return dict(row) if row else None

    def preview_import(
        self,
        dataset_id: str,
        data_type: str,
        file_name: str,
        content: bytes,
        mapping: dict[str, str] | None = None,
        overwrite_manual: bool = False,
    ) -> ImportPreview:
        if self.get_dataset(dataset_id) is None:
            raise ValueError("找不到该学情数据集")
        if data_type not in DATA_TYPE_SPECS:
            raise ValueError("未知学情数据类型")
        table = self.read_csv(content)
        selected_mapping = mapping or self.suggest_mapping(
            dataset_id,
            data_type,
            table.headers,
        )
        preview = ImportPreview(
            dataset_id=dataset_id,
            data_type=data_type,
            file_name=Path(file_name).name,
            content=content,
            headers=table.headers,
            mapping=selected_mapping,
        )
        preview.errors.extend(
            self.mapping_errors(data_type, selected_mapping)
        )
        if preview.errors:
            return preview

        spec = DATA_TYPE_SPECS[data_type]
        existing = self._existing_rows(dataset_id, data_type)
        seen_keys: set[tuple[Any, ...]] = set()
        seen_events: set[tuple[Any, ...]] = set()
        counts = Counter(
            {
                "insert": 0,
                "update": 0,
                "skip": 0,
                "manual_conflict": 0,
            }
        )
        for row_number, raw_row in table.rows:
            values: dict[str, Any] = {}
            extras: dict[str, str] = {}
            row_errors: list[str] = []
            for header, raw_value in raw_row.items():
                target = selected_mapping.get(header, EXTRA_FIELD)
                if target == IGNORE_FIELD:
                    continue
                if target == EXTRA_FIELD:
                    extras[header] = raw_value
                    continue
                field_spec = spec.field_by_key[target]
                try:
                    values[target] = self._coerce(raw_value, field_spec)
                except ValueError as exc:
                    row_errors.append(f"第 {row_number} 行：{exc}")
            if row_errors:
                preview.errors.extend(row_errors)
                continue
            canonical = {
                item.key: values.get(item.key) for item in spec.fields
            }
            key = tuple(canonical.get(item) for item in spec.natural_key)
            if any(value in (None, "") for value in key):
                preview.errors.append(
                    f"第 {row_number} 行：业务主键不能为空"
                )
                continue
            if key in seen_keys:
                preview.errors.append(
                    f"第 {row_number} 行：文件内业务主键重复 {key}"
                )
                continue
            seen_keys.add(key)
            if spec.extra_unique_key:
                event_key = tuple(
                    canonical.get(item) for item in spec.extra_unique_key
                )
                if event_key in seen_events:
                    preview.errors.append(
                        f"第 {row_number} 行：文件内作答事件版本重复 {event_key}"
                    )
                    continue
                seen_events.add(event_key)
                conflict = self._event_conflict(dataset_id, canonical)
                if (
                    conflict
                    and conflict.get("record_code") != canonical.get("record_code")
                ):
                    preview.errors.append(
                        f"第 {row_number} 行：作答事件版本已被记录 "
                        f"{conflict['record_code']} 使用"
                    )
                    continue

            issue_rows = self._business_issues(
                data_type,
                canonical,
                row_number,
            )
            preview.issues.extend(issue_rows)
            current = existing.get(key)
            action = "insert"
            existing_row_id: int | None = None
            if current is None:
                action = "insert"
            else:
                existing_row_id = int(current["row_id"])
                changed = any(
                    current.get(item.key) != canonical.get(item.key)
                    for item in spec.fields
                )
                if not changed:
                    action = "restore" if current.get("deleted_at") else "skip"
                elif int(current.get("manual_revision") or 0) > 0:
                    action = "update" if overwrite_manual else "manual_conflict"
                else:
                    action = "update"
            counts[
                "update"
                if action in {"update", "restore"}
                else action
            ] += 1
            row = {
                **canonical,
                "_row_number": row_number,
                "_action": action,
                "_existing_row_id": existing_row_id,
                "_source_json": raw_row,
                "_extra_json": extras,
            }
            preview.rows.append(row)
            preview.extras.append(extras)
        preview.counts = {
            "insert": counts["insert"],
            "update": counts["update"],
            "skip": counts["skip"],
            "manual_conflict": counts["manual_conflict"],
            "total": len(table.rows),
            "issues": len(preview.issues),
        }
        return preview

    def _write_raw_import(
        self,
        dataset_id: str,
        batch_id: str,
        file_name: str,
        content: bytes,
    ) -> tuple[str, Path]:
        safe_name = Path(file_name).name or "import.csv"
        relative = (
            Path("learning")
            / dataset_id
            / "imports"
            / batch_id
            / safe_name
        )
        destination = self.db_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return str(relative), destination

    def _log_change(
        self,
        connection: Any,
        *,
        dataset_id: str,
        data_type: str,
        row_id: int,
        action: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        change_source: str,
        import_batch_id: str | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO learning_change_log
            (dataset_id, data_type, row_id, action, before_json, after_json,
             change_source, import_batch_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                data_type,
                row_id,
                action,
                json.dumps(before, ensure_ascii=False) if before else None,
                json.dumps(after, ensure_ascii=False) if after else None,
                change_source,
                import_batch_id,
                utc_now(),
            ),
        )

    def execute_import(
        self,
        preview: ImportPreview,
        *,
        remember_mapping: bool = False,
    ) -> ImportResult:
        if preview.errors:
            raise ValueError("导入预览仍有错误，不能写入")
        spec = DATA_TYPE_SPECS[preview.data_type]
        batch_id = new_batch_id()
        content_hash = hashlib.sha256(preview.content).hexdigest()
        raw_path, raw_destination = self._write_raw_import(
            preview.dataset_id,
            batch_id,
            preview.file_name,
            preview.content,
        )
        now = utc_now()
        inserted = 0
        updated = 0
        skipped = 0
        conflicts = 0
        fields = [item.key for item in spec.fields]
        try:
            with self.repo.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO learning_import_batches
                    (id, dataset_id, data_type, file_name, sha256, raw_path,
                     status, total_rows, issue_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
                    """,
                    (
                        batch_id,
                        preview.dataset_id,
                        preview.data_type,
                        preview.file_name,
                        content_hash,
                        raw_path,
                        len(preview.rows),
                        len(preview.issues),
                        now,
                    ),
                )
                for row in preview.rows:
                    action = row["_action"]
                    if action in {"skip", "manual_conflict"}:
                        skipped += 1
                        conflicts += int(action == "manual_conflict")
                        continue
                    source_json = json.dumps(
                        row["_source_json"],
                        ensure_ascii=False,
                    )
                    extra_json = json.dumps(
                        row["_extra_json"],
                        ensure_ascii=False,
                    )
                    existing_row_id = row.get("_existing_row_id")
                    if existing_row_id is None:
                        columns = [
                            "dataset_id",
                            *fields,
                            "source_json",
                            "extra_json",
                            "import_batch_id",
                            "manual_revision",
                            "created_at",
                            "updated_at",
                        ]
                        values = [
                            preview.dataset_id,
                            *(row.get(key) for key in fields),
                            source_json,
                            extra_json,
                            batch_id,
                            0,
                            now,
                            now,
                        ]
                        cursor = connection.execute(
                            f"""
                            INSERT INTO {spec.table_name}
                            ({", ".join(columns)})
                            VALUES ({", ".join("?" for _ in columns)})
                            """,
                            values,
                        )
                        row_id = int(cursor.lastrowid)
                        inserted += 1
                        before = None
                        after = {
                            **{key: row.get(key) for key in fields},
                            "row_id": row_id,
                        }
                    else:
                        row_id = int(existing_row_id)
                        before_row = self._get_record_by_id(
                            connection,
                            spec,
                            preview.dataset_id,
                            row_id,
                        )
                        assignments = ", ".join(
                            [f"{key}=?" for key in fields]
                            + [
                                "source_json=?",
                                "extra_json=?",
                                "import_batch_id=?",
                                "manual_revision=0",
                                "updated_at=?",
                                "deleted_at=NULL",
                            ]
                        )
                        connection.execute(
                            f"""
                            UPDATE {spec.table_name}
                            SET {assignments}
                            WHERE row_id=? AND dataset_id=?
                            """,
                            (
                                *(row.get(key) for key in fields),
                                source_json,
                                extra_json,
                                batch_id,
                                now,
                                row_id,
                                preview.dataset_id,
                            ),
                        )
                        updated += 1
                        before = before_row
                        after = {
                            **{key: row.get(key) for key in fields},
                            "row_id": row_id,
                        }
                    self._log_change(
                        connection,
                        dataset_id=preview.dataset_id,
                        data_type=preview.data_type,
                        row_id=row_id,
                        action=action,
                        before=before,
                        after=after,
                        change_source="import",
                        import_batch_id=batch_id,
                    )
                for issue in preview.issues:
                    connection.execute(
                        """
                        INSERT INTO learning_import_issues
                        (dataset_id, batch_id, data_type, row_number, severity,
                         code, field_name, message, raw_value)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            preview.dataset_id,
                            batch_id,
                            preview.data_type,
                            issue.get("row_number"),
                            issue.get("severity", "warning"),
                            issue.get("code", "import_issue"),
                            issue.get("field_name"),
                            issue.get("message", ""),
                            str(issue.get("raw_value", "")),
                        ),
                    )
                if remember_mapping:
                    for source_name, target in preview.mapping.items():
                        if target in {EXTRA_FIELD, IGNORE_FIELD}:
                            continue
                        connection.execute(
                            """
                            INSERT INTO learning_field_aliases
                            (dataset_id, data_type, source_name, target_field,
                             created_at)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(dataset_id, data_type, source_name)
                            DO UPDATE SET target_field=excluded.target_field,
                                          created_at=excluded.created_at
                            """,
                            (
                                preview.dataset_id,
                                preview.data_type,
                                normalize_header(source_name),
                                target,
                                now,
                            ),
                        )
                connection.execute(
                    """
                    UPDATE learning_import_batches
                    SET status='completed', inserted_rows=?,
                        updated_rows=?, skipped_rows=?, conflict_rows=?,
                        finished_at=?
                    WHERE id=?
                    """,
                    (
                        inserted,
                        updated,
                        skipped,
                        conflicts,
                        utc_now(),
                        batch_id,
                    ),
                )
        except Exception:
            raw_destination.unlink(missing_ok=True)
            raise
        return ImportResult(
            batch_id=batch_id,
            inserted=inserted,
            updated=updated,
            skipped=skipped,
            conflicts=conflicts,
            issues=len(preview.issues),
        )

    def _get_record_by_id(
        self,
        connection: Any,
        spec: DataTypeSpec,
        dataset_id: str,
        row_id: int,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            f"""
            SELECT * FROM {spec.table_name}
            WHERE dataset_id=? AND row_id=?
            """,
            (dataset_id, row_id),
        ).fetchone()
        return dict(row) if row else None

    def list_records(
        self,
        dataset_id: str,
        data_type: str,
        *,
        include_deleted: bool = False,
        latest_answers: bool = False,
    ) -> list[dict[str, Any]]:
        spec = DATA_TYPE_SPECS[data_type]
        where = "" if include_deleted else "AND deleted_at IS NULL"
        with self.repo.connect() as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    f"""
                    SELECT * FROM {spec.table_name}
                    WHERE dataset_id=? {where}
                    ORDER BY row_id
                    """,
                    (dataset_id,),
                ).fetchall()
            ]
        if data_type == "answer" and latest_answers:
            latest: dict[str, dict[str, Any]] = {}
            for row in rows:
                event = str(row["answer_event_code"])
                current = latest.get(event)
                if current is None or int(row["revision"] or 0) > int(
                    current["revision"] or 0
                ):
                    latest[event] = row
            rows = sorted(latest.values(), key=lambda row: row["row_id"])
        return rows

    def get_record(
        self,
        dataset_id: str,
        data_type: str,
        row_id: int,
    ) -> dict[str, Any] | None:
        spec = DATA_TYPE_SPECS[data_type]
        with self.repo.connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM {spec.table_name}
                WHERE dataset_id=? AND row_id=?
                """,
                (dataset_id, row_id),
            ).fetchone()
        return dict(row) if row else None

    def _validated_values(
        self,
        data_type: str,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        spec = DATA_TYPE_SPECS[data_type]
        canonical: dict[str, Any] = {}
        for item in spec.fields:
            canonical[item.key] = self._coerce(values.get(item.key), item)
        missing = [
            item.label
            for item in spec.fields
            if item.required and canonical.get(item.key) in (None, "")
        ]
        if missing:
            raise ValueError("以下必填字段为空：" + "、".join(missing))
        key = tuple(canonical.get(item) for item in spec.natural_key)
        if any(value in (None, "") for value in key):
            raise ValueError("业务主键不能为空")
        return canonical

    def _ensure_unique(
        self,
        connection: Any,
        spec: DataTypeSpec,
        dataset_id: str,
        values: dict[str, Any],
        exclude_row_id: int | None = None,
    ) -> None:
        clauses = " AND ".join(f"{item}=?" for item in spec.natural_key)
        params = [dataset_id, *(values.get(item) for item in spec.natural_key)]
        query = (
            f"SELECT row_id FROM {spec.table_name} "
            f"WHERE dataset_id=? AND {clauses}"
        )
        if exclude_row_id is not None:
            query += " AND row_id<>?"
            params.append(exclude_row_id)
        duplicate = connection.execute(query, params).fetchone()
        if duplicate:
            raise ValueError("同一学情数据集内业务主键重复")
        if spec.extra_unique_key:
            clauses = " AND ".join(
                f"{item}=?" for item in spec.extra_unique_key
            )
            params = [
                dataset_id,
                *(values.get(item) for item in spec.extra_unique_key),
            ]
            query = (
                f"SELECT row_id FROM {spec.table_name} "
                f"WHERE dataset_id=? AND {clauses}"
            )
            if exclude_row_id is not None:
                query += " AND row_id<>?"
                params.append(exclude_row_id)
            duplicate = connection.execute(query, params).fetchone()
            if duplicate:
                raise ValueError("作答事件编号与修订版本组合重复")

    def _insert_record(
        self,
        connection: Any,
        dataset_id: str,
        data_type: str,
        values: dict[str, Any],
        *,
        change_source: str,
        manual_revision: int,
    ) -> int:
        spec = DATA_TYPE_SPECS[data_type]
        now = utc_now()
        fields = [item.key for item in spec.fields]
        source_json = json.dumps(values, ensure_ascii=False)
        columns = [
            "dataset_id",
            *fields,
            "source_json",
            "extra_json",
            "manual_revision",
            "created_at",
            "updated_at",
        ]
        cursor = connection.execute(
            f"""
            INSERT INTO {spec.table_name}
            ({", ".join(columns)})
            VALUES ({", ".join("?" for _ in columns)})
            """,
            (
                dataset_id,
                *(values.get(key) for key in fields),
                source_json,
                "{}",
                manual_revision,
                now,
                now,
            ),
        )
        row_id = int(cursor.lastrowid)
        after = {**values, "row_id": row_id}
        self._log_change(
            connection,
            dataset_id=dataset_id,
            data_type=data_type,
            row_id=row_id,
            action="create",
            before=None,
            after=after,
            change_source=change_source,
        )
        return row_id

    def create_record(
        self,
        dataset_id: str,
        data_type: str,
        values: dict[str, Any],
    ) -> int:
        if self.get_dataset(dataset_id) is None:
            raise ValueError("找不到该学情数据集")
        spec = DATA_TYPE_SPECS[data_type]
        canonical = self._validated_values(data_type, values)
        issues = self._business_issues(data_type, canonical, 0)
        if any(
            issue["code"] in {"unknown_data_table"}
            for issue in issues
        ):
            raise ValueError(
                "字段说明的数据表必须是已支持的六类数据之一"
            )
        with self.repo.connect() as connection:
            self._ensure_unique(
                connection,
                spec,
                dataset_id,
                canonical,
            )
            return self._insert_record(
                connection,
                dataset_id,
                data_type,
                canonical,
                change_source="manual",
                manual_revision=1,
            )

    def _cascade_code_change(
        self,
        connection: Any,
        *,
        dataset_id: str,
        data_type: str,
        old_values: dict[str, Any],
        new_values: dict[str, Any],
    ) -> None:
        now = utc_now()
        if data_type == "student" and (
            old_values.get("student_code") != new_values.get("student_code")
        ):
            related = connection.execute(
                """
                SELECT * FROM learning_answers
                WHERE dataset_id=? AND student_code=?
                """,
                (
                    dataset_id,
                    old_values.get("student_code"),
                ),
            ).fetchall()
            for item in related:
                connection.execute(
                    """
                    UPDATE learning_answers
                    SET student_code=?, manual_revision=manual_revision+1,
                        updated_at=?
                    WHERE row_id=?
                    """,
                    (
                        new_values.get("student_code"),
                        now,
                        item["row_id"],
                    ),
                )
                self._log_change(
                    connection,
                    dataset_id=dataset_id,
                    data_type="answer",
                    row_id=int(item["row_id"]),
                    action="cascade_identity",
                    before=dict(item),
                    after={
                        **dict(item),
                        "student_code": new_values.get("student_code"),
                    },
                    change_source="manual:cascade",
                )
        elif data_type == "material" and (
            old_values.get("material_code") != new_values.get("material_code")
        ):
            related = connection.execute(
                """
                SELECT * FROM learning_answers
                WHERE dataset_id=? AND material_code=?
                """,
                (
                    dataset_id,
                    old_values.get("material_code"),
                ),
            ).fetchall()
            for item in related:
                connection.execute(
                    """
                    UPDATE learning_answers
                    SET material_code=?, manual_revision=manual_revision+1,
                        updated_at=?
                    WHERE row_id=?
                    """,
                    (
                        new_values.get("material_code"),
                        now,
                        item["row_id"],
                    ),
                )
                self._log_change(
                    connection,
                    dataset_id=dataset_id,
                    data_type="answer",
                    row_id=int(item["row_id"]),
                    action="cascade_identity",
                    before=dict(item),
                    after={
                        **dict(item),
                        "material_code": new_values.get("material_code"),
                    },
                    change_source="manual:cascade",
                )
        elif data_type == "node" and (
            old_values.get("node_code") != new_values.get("node_code")
        ):
            related = connection.execute(
                """
                SELECT * FROM learning_answers
                WHERE dataset_id=? AND node_code=?
                """,
                (
                    dataset_id,
                    old_values.get("node_code"),
                ),
            ).fetchall()
            for item in related:
                connection.execute(
                    """
                    UPDATE learning_answers
                    SET node_code=?, manual_revision=manual_revision+1,
                        updated_at=?
                    WHERE row_id=?
                    """,
                    (
                        new_values.get("node_code"),
                        now,
                        item["row_id"],
                    ),
                )
                self._log_change(
                    connection,
                    dataset_id=dataset_id,
                    data_type="answer",
                    row_id=int(item["row_id"]),
                    action="cascade_identity",
                    before=dict(item),
                    after={
                        **dict(item),
                        "node_code": new_values.get("node_code"),
                    },
                    change_source="manual:cascade",
                )
        elif data_type == "requirement" and (
            old_values.get("indicator_code")
            != new_values.get("indicator_code")
        ):
            old_code = str(old_values.get("indicator_code") or "")
            new_code = str(new_values.get("indicator_code") or "")
            rows = connection.execute(
                """
                SELECT row_id, indicator_codes FROM learning_materials
                WHERE dataset_id=? AND indicator_codes IS NOT NULL
                """,
                (dataset_id,),
            ).fetchall()
            for row in rows:
                tokens = split_multi_value(row["indicator_codes"])
                if old_code not in tokens:
                    continue
                updated_tokens = [
                    new_code if token == old_code else token for token in tokens
                ]
                connection.execute(
                    """
                    UPDATE learning_materials
                    SET indicator_codes=?, manual_revision=manual_revision+1,
                        updated_at=?
                    WHERE row_id=?
                    """,
                    ("、".join(updated_tokens), now, row["row_id"]),
                )
                self._log_change(
                    connection,
                    dataset_id=dataset_id,
                    data_type="material",
                    row_id=int(row["row_id"]),
                    action="cascade_identity",
                    before=dict(row),
                    after={
                        **dict(row),
                        "indicator_codes": "、".join(updated_tokens),
                    },
                    change_source="manual:cascade",
                )

    def update_record(
        self,
        dataset_id: str,
        data_type: str,
        row_id: int,
        values: dict[str, Any],
        *,
        allow_identity_change: bool = False,
    ) -> None:
        spec = DATA_TYPE_SPECS[data_type]
        current = self.get_record(dataset_id, data_type, row_id)
        if current is None or current.get("deleted_at"):
            raise ValueError("找不到可修改的记录")
        merged = {
            item.key: (
                values[item.key]
                if item.key in values
                else current.get(item.key)
            )
            for item in spec.fields
        }
        canonical = self._validated_values(data_type, merged)
        validation_issues = self._business_issues(
            data_type,
            canonical,
            0,
        )
        if any(
            issue["code"] == "unknown_data_table"
            for issue in validation_issues
        ):
            raise ValueError(
                "字段说明的数据表必须是已支持的六类数据之一"
            )
        identity_changed = any(
            current.get(key) != canonical.get(key)
            for key in spec.natural_key
        )
        if identity_changed and not allow_identity_change:
            raise ValueError("业务编号必须通过“修改业务编号”操作变更")
        with self.repo.connect() as connection:
            self._ensure_unique(
                connection,
                spec,
                dataset_id,
                canonical,
                exclude_row_id=row_id,
            )
            assignments = ", ".join(
                [f"{item.key}=?" for item in spec.fields]
                + ["manual_revision=manual_revision+1", "updated_at=?"]
            )
            connection.execute(
                f"""
                UPDATE {spec.table_name}
                SET {assignments}
                WHERE dataset_id=? AND row_id=?
                """,
                (
                    *(canonical.get(item.key) for item in spec.fields),
                    utc_now(),
                    dataset_id,
                    row_id,
                ),
            )
            self._cascade_code_change(
                connection,
                dataset_id=dataset_id,
                data_type=data_type,
                old_values=current,
                new_values=canonical,
            )
            self._log_change(
                connection,
                dataset_id=dataset_id,
                data_type=data_type,
                row_id=row_id,
                action=(
                    "change_identity" if identity_changed else "update"
                ),
                before=current,
                after={**canonical, "row_id": row_id},
                change_source="manual",
            )

    def related_counts(
        self,
        dataset_id: str,
        data_type: str,
        row: dict[str, Any],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self.repo.connect() as connection:
            if data_type == "student":
                counts["作答记录"] = connection.execute(
                    """
                    SELECT COUNT(*) FROM learning_answers
                    WHERE dataset_id=? AND student_code=? AND deleted_at IS NULL
                    """,
                    (dataset_id, row.get("student_code")),
                ).fetchone()[0]
            elif data_type == "material":
                counts["作答记录"] = connection.execute(
                    """
                    SELECT COUNT(*) FROM learning_answers
                    WHERE dataset_id=? AND material_code=? AND deleted_at IS NULL
                    """,
                    (dataset_id, row.get("material_code")),
                ).fetchone()[0]
            elif data_type == "node":
                counts["作答记录"] = connection.execute(
                    """
                    SELECT COUNT(*) FROM learning_answers
                    WHERE dataset_id=? AND node_code=? AND deleted_at IS NULL
                    """,
                    (dataset_id, row.get("node_code")),
                ).fetchone()[0]
        return counts

    def _cascade_delete(
        self,
        connection: Any,
        dataset_id: str,
        data_type: str,
        row: dict[str, Any],
        *,
        deleted_at: str | None,
        action: str,
    ) -> int:
        column: str | None = None
        value: Any = None
        if data_type == "student":
            column, value = "student_code", row.get("student_code")
        elif data_type == "material":
            column, value = "material_code", row.get("material_code")
        elif data_type == "node":
            column, value = "node_code", row.get("node_code")
        if not column:
            return 0
        deleted_filter = (
            "deleted_at IS NOT NULL"
            if deleted_at is None
            else "deleted_at IS NULL"
        )
        related = connection.execute(
            f"""
            SELECT * FROM learning_answers
            WHERE dataset_id=? AND {column}=? AND {deleted_filter}
            """,
            (dataset_id, value),
        ).fetchall()
        for item in related:
            connection.execute(
                """
                UPDATE learning_answers
                SET deleted_at=?, updated_at=?
                WHERE row_id=?
                """,
                (deleted_at, utc_now(), item["row_id"]),
            )
            self._log_change(
                connection,
                dataset_id=dataset_id,
                data_type="answer",
                row_id=int(item["row_id"]),
                action=action,
                before=dict(item),
                after={
                    **dict(item),
                    "deleted_at": deleted_at,
                },
                change_source="manual:cascade",
            )
        return len(related)

    def soft_delete_record(
        self,
        dataset_id: str,
        data_type: str,
        row_id: int,
        *,
        cascade: bool = False,
    ) -> int:
        spec = DATA_TYPE_SPECS[data_type]
        current = self.get_record(dataset_id, data_type, row_id)
        if current is None or current.get("deleted_at"):
            raise ValueError("找不到可删除的记录")
        now = utc_now()
        with self.repo.connect() as connection:
            connection.execute(
                f"""
                UPDATE {spec.table_name} SET deleted_at=?, updated_at=?
                WHERE dataset_id=? AND row_id=?
                """,
                (now, now, dataset_id, row_id),
            )
            related = (
                self._cascade_delete(
                    connection,
                    dataset_id,
                    data_type,
                    current,
                    deleted_at=now,
                    action="soft_delete",
                )
                if cascade
                else 0
            )
            self._log_change(
                connection,
                dataset_id=dataset_id,
                data_type=data_type,
                row_id=row_id,
                action="soft_delete",
                before=current,
                after={**current, "deleted_at": now},
                change_source="manual",
            )
        return 1 + related

    def restore_record(
        self,
        dataset_id: str,
        data_type: str,
        row_id: int,
        *,
        cascade: bool = False,
    ) -> int:
        spec = DATA_TYPE_SPECS[data_type]
        current = self.get_record(dataset_id, data_type, row_id)
        if current is None or not current.get("deleted_at"):
            raise ValueError("找不到可恢复的记录")
        now = utc_now()
        with self.repo.connect() as connection:
            connection.execute(
                f"""
                UPDATE {spec.table_name}
                SET deleted_at=NULL, updated_at=?
                WHERE dataset_id=? AND row_id=?
                """,
                (now, dataset_id, row_id),
            )
            related = (
                self._cascade_delete(
                    connection,
                    dataset_id,
                    data_type,
                    current,
                    deleted_at=None,
                    action="restore",
                )
                if cascade
                else 0
            )
            self._log_change(
                connection,
                dataset_id=dataset_id,
                data_type=data_type,
                row_id=row_id,
                action="restore",
                before=current,
                after={**current, "deleted_at": None},
                change_source="manual",
            )
        return 1 + related

    def record_label(self, data_type: str, row: dict[str, Any]) -> str:
        spec = DATA_TYPE_SPECS[data_type]
        parts = [
            str(row.get(key))
            for key in spec.natural_key
            if row.get(key) not in (None, "")
        ]
        return " / ".join(parts) or f"#{row.get('row_id')}"

    def export_csv(
        self,
        dataset_id: str,
        data_type: str,
        *,
        rows: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[bytes, str]:
        spec = DATA_TYPE_SPECS[data_type]
        selected = (
            list(rows)
            if rows is not None
            else self.list_records(dataset_id, data_type)
        )
        extra_names: list[str] = []
        decoded_extras: list[dict[str, Any]] = []
        for row in selected:
            try:
                extra = json.loads(row.get("extra_json") or "{}")
            except json.JSONDecodeError:
                extra = {}
            decoded_extras.append(extra)
            for key in extra:
                if key not in extra_names:
                    extra_names.append(key)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            [item.label for item in spec.fields] + extra_names
        )
        for row, extra in zip(selected, decoded_extras, strict=True):
            writer.writerow(
                [
                    "" if row.get(item.key) is None else row.get(item.key)
                    for item in spec.fields
                ]
                + [extra.get(key, "") for key in extra_names]
            )
        dataset = self.get_dataset(dataset_id)
        safe_dataset = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", dataset["name"])
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_dataset}_{spec.label}_{stamp}.csv"
        return output.getvalue().encode("utf-8-sig"), filename

    def list_import_batches(
        self,
        dataset_id: str,
        data_type: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT * FROM learning_import_batches
            WHERE dataset_id=?
        """
        params: list[Any] = [dataset_id]
        if data_type:
            query += " AND data_type=?"
            params.append(data_type)
        query += " ORDER BY created_at DESC"
        with self.repo.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(query, params).fetchall()
            ]

    def quality_report(self, dataset_id: str) -> dict[str, Any]:
        all_records = {
            data_type: self.list_records(dataset_id, data_type)
            for data_type in DATA_TYPE_SPECS
        }
        issues: list[dict[str, Any]] = []

        def add_issue(
            data_type: str,
            row: dict[str, Any],
            field_name: str,
            code: str,
            message: str,
        ) -> None:
            issues.append(
                {
                    "数据类型": DATA_TYPE_SPECS[data_type].label,
                    "记录": self.record_label(data_type, row),
                    "字段": DATA_TYPE_SPECS[data_type].label_by_key.get(
                        field_name,
                        field_name,
                    ),
                    "问题代码": code,
                    "说明": message,
                    "row_id": row.get("row_id"),
                }
            )

        for data_type, records in all_records.items():
            spec = DATA_TYPE_SPECS[data_type]
            key_counts = Counter(
                tuple(row.get(key) for key in spec.natural_key)
                for row in records
            )
            for row in records:
                for item in spec.fields:
                    if item.required and row.get(item.key) in (None, ""):
                        add_issue(
                            data_type,
                            row,
                            item.key,
                            "blank_required",
                            f"{item.label}为空",
                        )
                key = tuple(row.get(key) for key in spec.natural_key)
                if key_counts[key] > 1:
                    add_issue(
                        data_type,
                        row,
                        spec.natural_key[0],
                        "duplicate_key",
                        "业务主键重复",
                    )

        students = {
            row["student_code"] for row in all_records["student"]
        }
        materials = {
            row["material_code"]: row for row in all_records["material"]
        }
        nodes = {
            row["node_code"]: row for row in all_records["node"]
        }
        themes = {
            row["theme_code"]
            for row in all_records["requirement"]
            if row.get("theme_code")
        }
        indicators = {
            row["indicator_code"]
            for row in all_records["requirement"]
            if row.get("indicator_code")
        }

        for row in all_records["material"]:
            for token in split_multi_value(row.get("indicator_codes")):
                if token not in indicators:
                    add_issue(
                        "material",
                        row,
                        "indicator_codes",
                        "missing_reference",
                        f"指标编号 {token} 在课程要求中不存在",
                    )
            if row.get("theme_code") and row["theme_code"] not in themes:
                add_issue(
                    "material",
                    row,
                    "theme_code",
                    "missing_reference",
                    f"主题编号 {row['theme_code']} 在课程要求中不存在",
                )
            if row.get("material_type") == "题目" and not row.get(
                "question_type"
            ):
                add_issue(
                    "material",
                    row,
                    "question_type",
                    "missing_question_type",
                    "题目材料缺少题型",
                )

        for row in all_records["node"]:
            for token in split_multi_value(row.get("taught_topics")):
                if token not in themes:
                    add_issue(
                        "node",
                        row,
                        "taught_topics",
                        "missing_reference",
                        f"主题编号 {token} 在课程要求中不存在",
                    )
            for token in split_multi_value(row.get("available_materials")):
                if token not in materials:
                    add_issue(
                        "node",
                        row,
                        "available_materials",
                        "missing_reference",
                        f"材料编号 {token} 在题目与材料中不存在",
                    )

        event_counts = Counter(
            (row.get("answer_event_code"), row.get("revision"))
            for row in all_records["answer"]
        )
        for row in all_records["answer"]:
            for field_name, pool, label in (
                ("student_code", students, "学生"),
                ("material_code", materials, "材料"),
                ("node_code", nodes, "节点"),
                ("theme_code", themes, "主题"),
            ):
                value = row.get(field_name)
                if value and value not in pool:
                    add_issue(
                        "answer",
                        row,
                        field_name,
                        "missing_reference",
                        f"{label}编号 {value} 不存在",
                    )
            event_key = (
                row.get("answer_event_code"),
                row.get("revision"),
            )
            if event_counts[event_key] > 1:
                add_issue(
                    "answer",
                    row,
                    "answer_event_code",
                    "duplicate_revision",
                    "作答事件编号与修订版本重复",
                )
            status = row.get("grading_status")
            score = row.get("score")
            full_score = row.get("full_score")
            if status not in GRADING_STATUSES:
                add_issue(
                    "answer",
                    row,
                    "grading_status",
                    "unknown_grading_status",
                    f"未知评分状态：{status}",
                )
            if score is not None and full_score is not None:
                if not 0 <= score <= full_score:
                    add_issue(
                        "answer",
                        row,
                        "score",
                        "score_out_of_range",
                        f"得分 {score} 不在 0—{full_score} 范围内",
                    )
            if status in {"待评分", "未作答"} and score is not None:
                add_issue(
                    "answer",
                    row,
                    "score",
                    "score_status_mismatch",
                    f"{status}记录的得分应为空",
                )
            if status == "已评分" and score is None:
                add_issue(
                    "answer",
                    row,
                    "score",
                    "score_status_mismatch",
                    "已评分记录的得分不能为空",
                )
            material = materials.get(row.get("material_code"))
            if (
                material
                and full_score is not None
                and material.get("full_score") is not None
                and full_score != material.get("full_score")
            ):
                add_issue(
                    "answer",
                    row,
                    "full_score",
                    "full_score_mismatch",
                    "作答满分与题目满分不一致",
                )
            rubric_scores = row.get("rubric_scores")
            if score is not None and rubric_scores:
                points = [
                    int(value)
                    for value in re.findall(r"=\s*(-?\d+)", str(rubric_scores))
                ]
                if points and sum(points) != score:
                    add_issue(
                        "answer",
                        row,
                        "rubric_scores",
                        "rubric_sum_mismatch",
                        f"评分点合计 {sum(points)} 与得分 {score} 不一致",
                    )

        for row in all_records["field_spec"]:
            if row.get("table_name") not in DATA_TYPES_BY_LABEL:
                add_issue(
                    "field_spec",
                    row,
                    "table_name",
                    "unknown_data_table",
                    f"未知数据表：{row.get('table_name')}",
                )

        status_counts = Counter(
            row.get("grading_status") for row in all_records["answer"]
        )
        return {
            "counts": {
                spec.label: len(all_records[data_type])
                for data_type, spec in DATA_TYPE_SPECS.items()
            },
            "issues": issues,
            "distributions": {
                "class": dict(
                    Counter(row.get("class_code") for row in all_records["student"])
                ),
                "grade": dict(
                    Counter(row.get("grade") for row in all_records["student"])
                ),
                "answer_node": dict(
                    Counter(row.get("node_code") for row in all_records["answer"])
                ),
                "answer_theme": dict(
                    Counter(row.get("theme_code") for row in all_records["answer"])
                ),
                "grading_status": dict(status_counts),
            },
        }

    def field_definition_rows(self, data_type: str) -> list[dict[str, Any]]:
        spec = DATA_TYPE_SPECS[data_type]
        return [
            {
                "字段名": item.label,
                "类型": TYPE_LABELS.get(item.value_type, item.value_type),
                "必填": "是" if item.required else "否",
                "说明": (
                    "可空" if not item.required else "不能为空"
                ),
            }
            for item in spec.fields
        ]

    def data_type_options(self) -> list[str]:
        return list(DATA_TYPE_SPECS)

    def valid_targets(self, data_type: str) -> list[str]:
        return [item.key for item in DATA_TYPE_SPECS[data_type].fields]

    def mapping_label(self, data_type: str, target: str) -> str:
        if target == EXTRA_FIELD:
            return "保留为扩展字段（不参与结构校验）"
        if target == IGNORE_FIELD:
            return "忽略此列"
        spec = DATA_TYPE_SPECS[data_type]
        return spec.label_by_key.get(target, target)

    def aliases(
        self,
        dataset_id: str,
        data_type: str,
    ) -> list[dict[str, Any]]:
        with self.repo.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT * FROM learning_field_aliases
                    WHERE dataset_id=? AND data_type=?
                    ORDER BY source_name
                    """,
                    (dataset_id, data_type),
                ).fetchall()
            ]

    def delete_alias(
        self,
        dataset_id: str,
        data_type: str,
        source_name: str,
    ) -> None:
        with self.repo.connect() as connection:
            connection.execute(
                """
                DELETE FROM learning_field_aliases
                WHERE dataset_id=? AND data_type=? AND source_name=?
                """,
                (dataset_id, data_type, source_name),
            )


def build_mapping_defaults(
    service: LearningDataService,
    dataset_id: str,
    data_type: str,
    headers: Iterable[str],
) -> dict[str, str]:
    return service.suggest_mapping(dataset_id, data_type, list(headers))


def rows_as_labels(
    data_type: str,
    rows: Sequence[dict[str, Any]],
) -> dict[int, str]:
    spec = DATA_TYPE_SPECS[data_type]
    result: dict[int, str] = {}
    for row in rows:
        parts = [
            str(row.get(key))
            for key in spec.natural_key
            if row.get(key) not in (None, "")
        ]
        if data_type == "answer":
            parts.append(f"v{row.get('revision')}")
        result[int(row["row_id"])] = " / ".join(parts)
    return result
