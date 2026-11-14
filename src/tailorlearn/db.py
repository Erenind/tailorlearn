from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from tailorlearn.models import (
    AIUnit,
    OutlineEvidence,
    OutlineItem,
    OutlineItemStatus,
    OutlineRelation,
    OutlineRelationType,
    OutlineRequirement,
    PageResult,
    UnitStatus,
)

REPOSITORY_API_VERSION = "2026-11-11-rename-material-slices"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS material_sets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    material_type TEXT NOT NULL,
    subject TEXT,
    stage TEXT,
    grade TEXT,
    semester TEXT,
    edition TEXT,
    notes TEXT,
    metadata_sources_json TEXT NOT NULL DEFAULT '{}',
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(name, material_type, version)
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    material_set_id TEXT NOT NULL REFERENCES material_sets(id),
    file_name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    order_index INTEGER NOT NULL DEFAULT 0,
    extraction_status TEXT NOT NULL DEFAULT 'pending',
    extraction_method TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(material_set_id, sha256)
);

CREATE TABLE IF NOT EXISTS pages (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    raw_text TEXT NOT NULL DEFAULT '',
    final_text TEXT NOT NULL DEFAULT '',
    source_kind TEXT NOT NULL,
    confidence REAL,
    status TEXT NOT NULL DEFAULT 'ok',
    warning_reason TEXT,
    char_count INTEGER NOT NULL DEFAULT 0,
    edited INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(document_id, page_number)
);

CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES sections(id) ON DELETE SET NULL,
    path TEXT NOT NULL,
    title TEXT NOT NULL,
    level INTEGER NOT NULL,
    order_index INTEGER NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    content TEXT NOT NULL,
    split_method TEXT NOT NULL DEFAULT 'heading-rules',
    created_at TEXT NOT NULL,
    UNIQUE(document_id, path)
);

CREATE TABLE IF NOT EXISTS units (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    unit_type TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    related_unit_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'draft',
    current_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS unit_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id TEXT NOT NULL REFERENCES units(id),
    version INTEGER NOT NULL,
    unit_type TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_refs_json TEXT NOT NULL,
    related_unit_ids_json TEXT NOT NULL,
    change_source TEXT NOT NULL,
    ai_profile TEXT,
    ai_base_url TEXT,
    ai_model TEXT,
    raw_response_path TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(unit_id, version)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    material_set_id TEXT REFERENCES material_sets(id),
    document_id TEXT REFERENCES documents(id),
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    total INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    message TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS job_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    item_key TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(job_id, item_key)
);

CREATE TABLE IF NOT EXISTS learning_datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS learning_import_batches (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    data_type TEXT NOT NULL,
    file_name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    raw_path TEXT NOT NULL,
    status TEXT NOT NULL,
    total_rows INTEGER NOT NULL DEFAULT 0,
    inserted_rows INTEGER NOT NULL DEFAULT 0,
    updated_rows INTEGER NOT NULL DEFAULT 0,
    skipped_rows INTEGER NOT NULL DEFAULT 0,
    conflict_rows INTEGER NOT NULL DEFAULT 0,
    issue_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS learning_field_aliases (
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    data_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    target_field TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(dataset_id, data_type, source_name)
);

CREATE TABLE IF NOT EXISTS learning_import_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    batch_id TEXT NOT NULL REFERENCES learning_import_batches(id),
    data_type TEXT NOT NULL,
    row_number INTEGER,
    severity TEXT NOT NULL,
    code TEXT NOT NULL,
    field_name TEXT,
    message TEXT NOT NULL,
    raw_value TEXT
);

CREATE TABLE IF NOT EXISTS learning_change_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    data_type TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT,
    change_source TEXT NOT NULL,
    import_batch_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_snapshots (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    scope_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    change_log_cutoff INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_diagnosis_jobs (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    class_code TEXT NOT NULL,
    node_code TEXT NOT NULL,
    change_log_cutoff INTEGER NOT NULL DEFAULT 0,
    provider_id TEXT NOT NULL,
    provider_vendor TEXT NOT NULL,
    provider_base_url TEXT NOT NULL,
    provider_model TEXT NOT NULL,
    status TEXT NOT NULL,
    total INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS ai_diagnosis_job_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES ai_diagnosis_jobs(id) ON DELETE CASCADE,
    student_code TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    diagnosis_id TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(job_id, student_code)
);

CREATE TABLE IF NOT EXISTS ai_student_diagnoses (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES ai_diagnosis_jobs(id) ON DELETE CASCADE,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    student_code TEXT NOT NULL,
    node_code TEXT NOT NULL,
    change_log_cutoff INTEGER NOT NULL DEFAULT 0,
    evidence_pack_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    provider_vendor TEXT NOT NULL,
    provider_base_url TEXT NOT NULL,
    provider_model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'needs_review',
    reviewed_summary TEXT NOT NULL DEFAULT '',
    review_note TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_teaching_recommendations (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    class_code TEXT NOT NULL,
    node_code TEXT NOT NULL,
    change_log_cutoff INTEGER NOT NULL DEFAULT 0,
    evidence_pack_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    provider_vendor TEXT NOT NULL,
    provider_base_url TEXT NOT NULL,
    provider_model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'needs_review',
    reviewed_summary TEXT NOT NULL DEFAULT '',
    review_note TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS teaching_chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_trace_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_teaching_chat_messages_created
    ON teaching_chat_messages(id);

CREATE TABLE IF NOT EXISTS teaching_chat_memory (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    covered_message_id INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS teaching_chat_suggestions (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    is_preset INTEGER NOT NULL DEFAULT 0,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS learning_students (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    student_code TEXT NOT NULL,
    class_code TEXT,
    grade INTEGER,
    source_json TEXT NOT NULL DEFAULT '{}',
    extra_json TEXT NOT NULL DEFAULT '{}',
    import_batch_id TEXT REFERENCES learning_import_batches(id),
    manual_revision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(dataset_id, student_code)
);

CREATE TABLE IF NOT EXISTS learning_requirements (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    theme_code TEXT,
    theme_name TEXT,
    theme_goal TEXT,
    indicator_code TEXT NOT NULL,
    indicator_name TEXT,
    standard_ref TEXT,
    standard_page TEXT,
    textbook_page TEXT,
    source_json TEXT NOT NULL DEFAULT '{}',
    extra_json TEXT NOT NULL DEFAULT '{}',
    import_batch_id TEXT REFERENCES learning_import_batches(id),
    manual_revision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(dataset_id, indicator_code)
);

CREATE TABLE IF NOT EXISTS learning_materials (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    material_code TEXT NOT NULL,
    theme_code TEXT,
    material_type TEXT,
    question_type TEXT,
    content TEXT,
    options TEXT,
    reference_answer TEXT,
    reference_solution TEXT,
    full_score INTEGER,
    rubric TEXT,
    indicator_codes TEXT,
    source TEXT,
    source_json TEXT NOT NULL DEFAULT '{}',
    extra_json TEXT NOT NULL DEFAULT '{}',
    import_batch_id TEXT REFERENCES learning_import_batches(id),
    manual_revision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(dataset_id, material_code)
);

CREATE TABLE IF NOT EXISTS learning_nodes (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    node_code TEXT NOT NULL,
    test_date TEXT,
    taught_topics TEXT,
    description TEXT,
    available_materials TEXT,
    source_json TEXT NOT NULL DEFAULT '{}',
    extra_json TEXT NOT NULL DEFAULT '{}',
    import_batch_id TEXT REFERENCES learning_import_batches(id),
    manual_revision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(dataset_id, node_code)
);

CREATE TABLE IF NOT EXISTS learning_answers (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    record_code TEXT NOT NULL,
    student_code TEXT,
    material_code TEXT,
    theme_code TEXT,
    node_code TEXT,
    answer_event_code TEXT NOT NULL,
    revision INTEGER NOT NULL,
    submitted_at TEXT,
    grading_status TEXT,
    score INTEGER,
    full_score INTEGER,
    final_answer TEXT,
    solution TEXT,
    rubric_scores TEXT,
    source_json TEXT NOT NULL DEFAULT '{}',
    extra_json TEXT NOT NULL DEFAULT '{}',
    import_batch_id TEXT REFERENCES learning_import_batches(id),
    manual_revision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(dataset_id, record_code),
    UNIQUE(dataset_id, answer_event_code, revision)
);

CREATE TABLE IF NOT EXISTS learning_field_definitions (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    table_name TEXT NOT NULL,
    field_name TEXT NOT NULL,
    field_type TEXT,
    meaning TEXT,
    allowed_format TEXT,
    example TEXT,
    blank_behavior TEXT,
    source_json TEXT NOT NULL DEFAULT '{}',
    extra_json TEXT NOT NULL DEFAULT '{}',
    import_batch_id TEXT REFERENCES learning_import_batches(id),
    manual_revision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(dataset_id, table_name, field_name)
);

CREATE INDEX IF NOT EXISTS idx_documents_set ON documents(material_set_id);
CREATE INDEX IF NOT EXISTS idx_sections_document ON sections(document_id, order_index);
CREATE INDEX IF NOT EXISTS idx_units_document ON units(document_id, section_id);
CREATE INDEX IF NOT EXISTS idx_units_status ON units(status, deleted_at);
CREATE INDEX IF NOT EXISTS idx_jobs_document ON jobs(document_id, started_at);
CREATE INDEX IF NOT EXISTS idx_learning_batches_dataset
    ON learning_import_batches(dataset_id, created_at);
CREATE INDEX IF NOT EXISTS idx_learning_changes_dataset
    ON learning_change_log(dataset_id, created_at);
CREATE INDEX IF NOT EXISTS idx_analysis_snapshots_dataset
    ON analysis_snapshots(dataset_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ai_diagnosis_jobs_dataset
    ON ai_diagnosis_jobs(dataset_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ai_diagnosis_job_items_job
    ON ai_diagnosis_job_items(job_id, status);
CREATE INDEX IF NOT EXISTS idx_ai_student_diagnoses_student
    ON ai_student_diagnoses(dataset_id, student_code, node_code, created_at);
CREATE INDEX IF NOT EXISTS idx_ai_teaching_recommendations_scope
    ON ai_teaching_recommendations(dataset_id, class_code, node_code, created_at);
CREATE INDEX IF NOT EXISTS idx_learning_students_dataset
    ON learning_students(dataset_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_learning_requirements_dataset
    ON learning_requirements(dataset_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_learning_materials_dataset
    ON learning_materials(dataset_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_learning_nodes_dataset
    ON learning_nodes(dataset_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_learning_answers_dataset
    ON learning_answers(dataset_id, deleted_at, answer_event_code, revision);
CREATE INDEX IF NOT EXISTS idx_learning_field_definitions_dataset
    ON learning_field_definitions(dataset_id, deleted_at);

CREATE TABLE IF NOT EXISTS outline_runs (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES learning_datasets(id),
    name TEXT NOT NULL,
    method TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    boundary_note TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft',
    item_count INTEGER NOT NULL DEFAULT 0,
    relation_count INTEGER NOT NULL DEFAULT 0,
    ai_vendor TEXT,
    ai_base_url TEXT,
    ai_model TEXT,
    raw_response_path TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS outline_items (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES outline_runs(id) ON DELETE CASCADE,
    item_key TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    theme_code TEXT,
    theme_name TEXT,
    item_type TEXT NOT NULL,
    content TEXT NOT NULL,
    key_points_json TEXT NOT NULL DEFAULT '[]',
    indicator_codes_json TEXT NOT NULL DEFAULT '[]',
    requirements_json TEXT NOT NULL DEFAULT '[]',
    sources_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT,
    review_warnings_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'draft',
    current_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(run_id, item_key)
);

CREATE TABLE IF NOT EXISTS outline_item_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL REFERENCES outline_items(id),
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    key_points_json TEXT NOT NULL,
    indicator_codes_json TEXT NOT NULL,
    requirements_json TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    notes TEXT,
    change_source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(item_id, version)
);

CREATE TABLE IF NOT EXISTS outline_relations (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES outline_runs(id) ON DELETE CASCADE,
    source_item_id TEXT NOT NULL REFERENCES outline_items(id) ON DELETE CASCADE,
    target_item_id TEXT NOT NULL REFERENCES outline_items(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    basis TEXT NOT NULL DEFAULT '',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    origin TEXT NOT NULL DEFAULT 'rule',
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(run_id, source_item_id, target_item_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_outline_runs_dataset
    ON outline_runs(dataset_id, created_at);
CREATE INDEX IF NOT EXISTS idx_outline_items_run
    ON outline_items(run_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_outline_relations_run
    ON outline_relations(run_id, created_at);
"""


class Repository:
    def __init__(self, database_file: Path):
        self.database_file = Path(database_file)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.database_file.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_file, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(material_sets)"
                ).fetchall()
            }
            if "metadata_sources_json" not in columns:
                connection.execute(
                    "ALTER TABLE material_sets "
                    "ADD COLUMN metadata_sources_json TEXT NOT NULL DEFAULT '{}'"
                )
            chat_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(teaching_chat_messages)"
                ).fetchall()
            }
            if "tool_trace_json" not in chat_columns:
                connection.execute(
                    "ALTER TABLE teaching_chat_messages "
                    "ADD COLUMN tool_trace_json TEXT NOT NULL DEFAULT '[]'"
                )
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", "9"),
            )

    def create_material_set(
        self,
        *,
        name: str,
        material_type: str,
        subject: str | None = None,
        stage: str | None = None,
        grade: str | None = None,
        semester: str | None = None,
        edition: str | None = None,
        notes: str | None = None,
        append_to: str | None = None,
    ) -> dict[str, Any]:
        import uuid

        name = name.strip()
        if not name:
            raise ValueError("材料名称不能为空")
        with self.connect() as connection:
            if append_to:
                row = connection.execute(
                    "SELECT * FROM material_sets WHERE id=? AND deleted_at IS NULL",
                    (append_to,),
                ).fetchone()
                if row is None:
                    raise ValueError("找不到要追加的材料")
                if row["material_type"] != material_type:
                    raise ValueError("不能在另一种材料类型中追加文件")
                return dict(row)
            version = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1
                FROM material_sets WHERE name=? AND material_type=?
                """,
                (name, material_type),
            ).fetchone()[0]
            material_id = f"set_{uuid.uuid4().hex[:12]}"
            provided_metadata = {
                "subject": subject,
                "stage": stage,
                "grade": grade,
                "semester": semester,
                "edition": edition,
                "notes": notes,
            }
            metadata_sources = {
                key: "user"
                for key, value in provided_metadata.items()
                if value
            }
            connection.execute(
                """
                INSERT INTO material_sets
                (id, name, material_type, subject, stage, grade, semester,
                 edition, notes, metadata_sources_json, version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    material_id,
                    name,
                    material_type,
                    subject or None,
                    stage or None,
                    grade or None,
                    semester or None,
                    edition or None,
                    notes or None,
                    json.dumps(metadata_sources, ensure_ascii=False),
                    version,
                    utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM material_sets WHERE id=?", (material_id,)
            ).fetchone()
            return dict(row)

    def add_document(
        self,
        *,
        document_id: str,
        material_set_id: str,
        file_name: str,
        sha256: str,
        stored_path: str,
        order_index: int,
        page_count: int,
        extraction_method: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM documents WHERE material_set_id=? AND sha256=?",
                (material_set_id, sha256),
            ).fetchone()
            if existing:
                if existing["deleted_at"] is None:
                    return dict(existing), False
                connection.execute(
                    """
                    UPDATE documents
                    SET file_name=?, stored_path=?, page_count=?,
                        order_index=?, extraction_method=?, deleted_at=NULL
                    WHERE id=?
                    """,
                    (
                        file_name,
                        stored_path,
                        page_count,
                        order_index,
                        extraction_method,
                        existing["id"],
                    ),
                )
                restored = connection.execute(
                    "SELECT * FROM documents WHERE id=?",
                    (existing["id"],),
                ).fetchone()
                return dict(restored), True
            connection.execute(
                """
                INSERT INTO documents
                (id, material_set_id, file_name, sha256, stored_path,
                 page_count, order_index, extraction_method, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    material_set_id,
                    file_name,
                    sha256,
                    stored_path,
                    page_count,
                    order_index,
                    extraction_method,
                    utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            return dict(row), True

    def update_document(
        self,
        document_id: str,
        **fields: Any,
    ) -> None:
        allowed = {
            "page_count",
            "extraction_status",
            "extraction_method",
            "metadata_json",
        }
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            return
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE documents SET {assignments} WHERE id=?",
                (*values.values(), document_id),
            )

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_documents(self, material_set_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM documents
                WHERE material_set_id=? AND deleted_at IS NULL
                ORDER BY order_index, file_name
                """,
                (material_set_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def rename_document(self, document_id: str, file_name: str) -> dict[str, Any]:
        clean_name = file_name.strip()
        if not clean_name:
            raise ValueError("切片名称不能为空")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET file_name=?
                WHERE id=? AND deleted_at IS NULL
                """,
                (clean_name, document_id),
            )
            if cursor.rowcount == 0:
                raise ValueError("该切片不存在或已被删除")
            row = connection.execute(
                "SELECT * FROM documents WHERE id=?",
                (document_id,),
            ).fetchone()
            return dict(row)

    def soft_delete_document(self, document_id: str) -> bool:
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET deleted_at=?
                WHERE id=? AND deleted_at IS NULL
                """,
                (now, document_id),
            )
            return cursor.rowcount > 0

    def list_material_sets(
        self,
        material_type: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT ms.*,
                   COUNT(d.id) AS document_count
            FROM material_sets ms
            LEFT JOIN documents d
              ON d.material_set_id=ms.id AND d.deleted_at IS NULL
            WHERE ms.deleted_at IS NULL
        """
        params: list[Any] = []
        if material_type:
            query += " AND ms.material_type=?"
            params.append(material_type)
        query += """
            GROUP BY ms.id
            ORDER BY ms.created_at DESC, ms.name
        """
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(query, params).fetchall()
            ]

    def get_material_set(self, material_set_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM material_sets WHERE id=?", (material_set_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_material_metadata(
        self,
        material_set_id: str,
        *,
        source: str = "user",
        **fields: Any,
    ) -> None:
        allowed = {"subject", "stage", "grade", "semester", "edition", "notes"}
        values = {
            key: (value.strip() if isinstance(value, str) else value) or None
            for key, value in fields.items()
            if key in allowed
        }
        if not values:
            return
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            current = connection.execute(
                "SELECT metadata_sources_json FROM material_sets WHERE id=?",
                (material_set_id,),
            ).fetchone()
            if current is None:
                raise ValueError("找不到材料")
            try:
                sources = json.loads(current["metadata_sources_json"] or "{}")
            except json.JSONDecodeError:
                sources = {}
            for key in values:
                sources[key] = source
            connection.execute(
                f"""
                UPDATE material_sets
                SET {assignments}, metadata_sources_json=?
                WHERE id=?
                """,
                (
                    *values.values(),
                    json.dumps(sources, ensure_ascii=False),
                    material_set_id,
                ),
            )

    def replace_pages(self, document_id: str, pages: Sequence[PageResult]) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("DELETE FROM pages WHERE document_id=?", (document_id,))
            connection.executemany(
                """
                INSERT INTO pages
                (document_id, page_number, raw_text, final_text, source_kind,
                 confidence, status, warning_reason, char_count, edited, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                [
                    (
                        document_id,
                        page.page_number,
                        page.raw_text,
                        page.final_text,
                        page.source_kind,
                        page.confidence,
                        page.status,
                        page.warning_reason,
                        page.char_count,
                        now,
                    )
                    for page in pages
                ],
            )

    def upsert_page(self, document_id: str, page: PageResult) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO pages
                (document_id, page_number, raw_text, final_text, source_kind,
                 confidence, status, warning_reason, char_count, edited, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(document_id, page_number) DO UPDATE SET
                    raw_text=excluded.raw_text,
                    final_text=excluded.final_text,
                    source_kind=excluded.source_kind,
                    confidence=excluded.confidence,
                    status=excluded.status,
                    warning_reason=excluded.warning_reason,
                    char_count=excluded.char_count,
                    updated_at=excluded.updated_at
                """,
                (
                    document_id,
                    page.page_number,
                    page.raw_text,
                    page.final_text,
                    page.source_kind,
                    page.confidence,
                    page.status,
                    page.warning_reason,
                    page.char_count,
                    utc_now(),
                ),
            )

    def list_pages(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM pages WHERE document_id=? ORDER BY page_number",
                (document_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def update_page_text(
        self,
        document_id: str,
        page_number: int,
        final_text: str,
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pages
                SET final_text=?, source_kind='edited', status='ok',
                    warning_reason=NULL, char_count=?, edited=1, updated_at=?
                WHERE document_id=? AND page_number=?
                """,
                (
                    final_text,
                    len(final_text),
                    utc_now(),
                    document_id,
                    page_number,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError("找不到该页")

    def replace_sections(
        self,
        document_id: str,
        sections: Sequence[Any],
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            existing_units = connection.execute(
                "SELECT COUNT(*) FROM units WHERE document_id=?", (document_id,)
            ).fetchone()[0]
            if existing_units:
                raise ValueError("该文档已有结构化单元，不能直接覆盖章节切分")
            connection.execute("DELETE FROM sections WHERE document_id=?", (document_id,))
            connection.executemany(
                """
                INSERT INTO sections
                (id, document_id, parent_id, path, title, level, order_index,
                 page_start, page_end, content, split_method, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        section.id,
                        document_id,
                        section.parent_id,
                        section.path,
                        section.title,
                        section.level,
                        section.order_index,
                        section.page_start,
                        section.page_end,
                        section.content,
                        section.split_method,
                        now,
                    )
                    for section in sections
                ],
            )

    def list_sections(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sections WHERE document_id=? ORDER BY order_index",
                (document_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_section(self, section_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM sections WHERE id=?", (section_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_section(
        self,
        *,
        section_id: str,
        title: str,
        content: str,
    ) -> None:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("章节标题不能为空")
        with self.connect() as connection:
            current = connection.execute(
                "SELECT * FROM sections WHERE id=?",
                (section_id,),
            ).fetchone()
            if current is None:
                raise ValueError("找不到该章节")
            sections = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT * FROM sections
                    WHERE document_id=?
                    ORDER BY order_index
                    """,
                    (current["document_id"],),
                ).fetchall()
            ]
            section_by_id = {section["id"]: section for section in sections}
            parent = section_by_id.get(current["parent_id"])
            new_path = (
                f"{parent['path']} / {clean_title}"
                if parent is not None
                else clean_title
            )
            old_path = current["path"]
            updated_paths: dict[str, str] = {}
            for section in sections:
                if section["id"] == section_id:
                    updated_paths[section_id] = new_path
                elif section["path"].startswith(f"{old_path} / "):
                    updated_paths[section["id"]] = (
                        new_path + section["path"][len(old_path) :]
                    )
            paths_by_id = {
                section["id"]: updated_paths.get(section["id"], section["path"])
                for section in sections
            }
            if len(set(paths_by_id.values())) != len(paths_by_id):
                raise ValueError("章节标题与同一文档中的其他章节冲突")
            for path_section_id, path in updated_paths.items():
                if path_section_id == section_id:
                    connection.execute(
                        """
                        UPDATE sections
                        SET title=?, path=?, content=?
                        WHERE id=?
                        """,
                        (clean_title, path, content, section_id),
                    )
                else:
                    connection.execute(
                        "UPDATE sections SET path=? WHERE id=?",
                        (path, path_section_id),
                    )

    def refresh_section_contents(self, document_id: str) -> int:
        pages = self.list_pages(document_id)
        sections = self.list_sections(document_id)
        updated = 0
        with self.connect() as connection:
            for section in sections:
                content = "\n\n".join(
                    f"【第 {page['page_number']} 页】\n{page['final_text']}"
                    for page in pages
                    if section["page_start"]
                    <= int(page["page_number"])
                    <= section["page_end"]
                    and str(page["final_text"]).strip()
                )
                connection.execute(
                    "UPDATE sections SET content=? WHERE id=?",
                    (content, section["id"]),
                )
                updated += 1
        return updated

    def save_units_for_section(
        self,
        *,
        document_id: str,
        section_id: str,
        units: Sequence[AIUnit],
        ai_profile: str,
        ai_base_url: str,
        ai_model: str,
        raw_response_path: str,
    ) -> list[str]:
        now = utc_now()
        saved_ids: list[str] = []
        with self.connect() as connection:
            existing_rows = connection.execute(
                "SELECT * FROM units WHERE section_id=?", (section_id,)
            ).fetchall()
            existing = {row["id"]: dict(row) for row in existing_rows}
            for index, unit in enumerate(units, start=1):
                unit_id = f"{section_id}_u{index:03d}"
                saved_ids.append(unit_id)
                current = existing.get(unit_id)
                version = (current["current_version"] + 1) if current else 1
                sources = json.dumps(
                    [ref.model_dump() for ref in unit.source_refs],
                    ensure_ascii=False,
                )
                related = json.dumps(unit.related_unit_ids, ensure_ascii=False)
                status = (
                    UnitStatus.NEEDS_REVIEW
                    if unit.review_warnings
                    else UnitStatus.DRAFT
                )
                connection.execute(
                    """
                    INSERT INTO units
                    (id, document_id, section_id, unit_type, title, content,
                     source_refs_json, related_unit_ids_json, status,
                     current_version, created_at, updated_at, deleted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    ON CONFLICT(id) DO UPDATE SET
                        unit_type=excluded.unit_type,
                        title=excluded.title,
                        content=excluded.content,
                        source_refs_json=excluded.source_refs_json,
                        related_unit_ids_json=excluded.related_unit_ids_json,
                        status=excluded.status,
                        current_version=excluded.current_version,
                        updated_at=excluded.updated_at,
                        deleted_at=NULL
                    """,
                    (
                        unit_id,
                        document_id,
                        section_id,
                        unit.unit_type,
                        unit.title,
                        unit.content,
                        sources,
                        related,
                        status,
                        version,
                        current["created_at"] if current else now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO unit_versions
                    (unit_id, version, unit_type, title, content,
                     source_refs_json, related_unit_ids_json, change_source,
                     ai_profile, ai_base_url, ai_model, raw_response_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'ai', ?, ?, ?, ?, ?)
                    """,
                    (
                        unit_id,
                        version,
                        unit.unit_type,
                        unit.title,
                        unit.content,
                        sources,
                        related,
                        ai_profile,
                        ai_base_url,
                        ai_model,
                        raw_response_path,
                        now,
                    ),
                )
            stale = set(existing) - set(saved_ids)
            if stale:
                placeholders = ",".join("?" for _ in stale)
                connection.execute(
                    f"UPDATE units SET deleted_at=? WHERE id IN ({placeholders})",
                    (now, *stale),
                )
        return saved_ids

    def list_units(
        self,
        *,
        document_id: str | None = None,
        material_set_id: str | None = None,
        include_deleted: bool = False,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT u.*, s.title AS section_title, s.path AS section_path,
                   d.file_name, d.material_set_id
            FROM units u
            JOIN sections s ON s.id=u.section_id
            JOIN documents d ON d.id=u.document_id
            WHERE 1=1
        """
        params: list[Any] = []
        if document_id:
            query += " AND u.document_id=?"
            params.append(document_id)
        if material_set_id:
            query += " AND d.material_set_id=?"
            params.append(material_set_id)
        if not include_deleted:
            query += " AND u.deleted_at IS NULL"
        if status:
            query += " AND u.status=?"
            params.append(status)
        query += " ORDER BY d.order_index, s.order_index, u.id"
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(query, params).fetchall()
            ]

    def get_unit(self, unit_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM units WHERE id=?", (unit_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_unit(
        self,
        *,
        unit_id: str,
        title: str,
        unit_type: str,
        content: str,
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            current = connection.execute(
                "SELECT * FROM units WHERE id=? AND deleted_at IS NULL",
                (unit_id,),
            ).fetchone()
            if current is None:
                raise ValueError("找不到该单元")
            version = current["current_version"] + 1
            connection.execute(
                """
                UPDATE units
                SET title=?, unit_type=?, content=?, current_version=?,
                    status='draft', updated_at=?
                WHERE id=?
                """,
                (title, unit_type, content, version, now, unit_id),
            )
            connection.execute(
                """
                INSERT INTO unit_versions
                (unit_id, version, unit_type, title, content,
                 source_refs_json, related_unit_ids_json, change_source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'manual', ?)
                """,
                (
                    unit_id,
                    version,
                    unit_type,
                    title,
                    content,
                    current["source_refs_json"],
                    current["related_unit_ids_json"],
                    now,
                ),
            )

    def set_units_status(self, unit_ids: Sequence[str], status: str) -> int:
        if status not in set(UnitStatus):
            raise ValueError("无效的审核状态")
        if not unit_ids:
            return 0
        placeholders = ",".join("?" for _ in unit_ids)
        with self.connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE units SET status=?, updated_at=?
                WHERE deleted_at IS NULL AND id IN ({placeholders})
                """,
                (status, utc_now(), *unit_ids),
            )
            return cursor.rowcount

    def soft_delete_units(self, unit_ids: Sequence[str]) -> int:
        if not unit_ids:
            return 0
        placeholders = ",".join("?" for _ in unit_ids)
        with self.connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE units SET deleted_at=?, updated_at=?
                WHERE deleted_at IS NULL AND id IN ({placeholders})
                """,
                (utc_now(), utc_now(), *unit_ids),
            )
            return cursor.rowcount

    def restore_units(self, unit_ids: Sequence[str]) -> int:
        if not unit_ids:
            return 0
        placeholders = ",".join("?" for _ in unit_ids)
        with self.connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE units SET deleted_at=NULL, updated_at=?
                WHERE id IN ({placeholders})
                """,
                (utc_now(), *unit_ids),
            )
            return cursor.rowcount

    def create_job(
        self,
        *,
        job_id: str,
        material_set_id: str | None,
        document_id: str | None,
        stage: str,
        total: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs
                (id, material_set_id, document_id, stage, status, total,
                 completed, failed, started_at)
                VALUES (?, ?, ?, ?, 'running', ?, 0, 0, ?)
                """,
                (job_id, material_set_id, document_id, stage, total, utc_now()),
            )

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        completed: int | None = None,
        failed: int | None = None,
        message: str | None = None,
        finish: bool = False,
    ) -> None:
        fields: dict[str, Any] = {}
        if status is not None:
            fields["status"] = status
        if completed is not None:
            fields["completed"] = completed
        if failed is not None:
            fields["failed"] = failed
        if message is not None:
            fields["message"] = message
        if finish:
            fields["finished_at"] = utc_now()
        if not fields:
            return
        assignments = ", ".join(f"{key}=?" for key in fields)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id=?",
                (*fields.values(), job_id),
            )

    def create_outline_run(
        self,
        *,
        run_id: str,
        dataset_id: str,
        name: str,
        method: str,
        scope: dict[str, Any],
        boundary_note: str = "",
        status: str = "draft",
        ai_vendor: str | None = None,
        ai_base_url: str | None = None,
        ai_model: str | None = None,
        raw_response_path: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO outline_runs
                (id, dataset_id, name, method, scope_json, boundary_note,
                 status, item_count, relation_count, ai_vendor, ai_base_url,
                 ai_model, raw_response_path, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    dataset_id,
                    name,
                    method,
                    json.dumps(scope, ensure_ascii=False),
                    boundary_note,
                    status,
                    ai_vendor,
                    ai_base_url,
                    ai_model,
                    raw_response_path,
                    notes,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM outline_runs WHERE id=?",
                (run_id,),
            ).fetchone()
            return dict(row)

    def update_outline_run(
        self,
        run_id: str,
        *,
        name: str | None = None,
        scope: dict[str, Any] | None = None,
        boundary_note: str | None = None,
        status: str | None = None,
        item_count: int | None = None,
        relation_count: int | None = None,
        raw_response_path: str | None = None,
        notes: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {}
        if name is not None:
            fields["name"] = name
        if scope is not None:
            fields["scope_json"] = json.dumps(scope, ensure_ascii=False)
        if boundary_note is not None:
            fields["boundary_note"] = boundary_note
        if status is not None:
            fields["status"] = status
        if item_count is not None:
            fields["item_count"] = item_count
        if relation_count is not None:
            fields["relation_count"] = relation_count
        if raw_response_path is not None:
            fields["raw_response_path"] = raw_response_path
        if notes is not None:
            fields["notes"] = notes
        if not fields:
            return
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key}=?" for key in fields)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE outline_runs SET {assignments} WHERE id=?",
                (*fields.values(), run_id),
            )

    def get_outline_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM outline_runs WHERE id=?",
                (run_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_outline_runs(
        self,
        dataset_id: str | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT r.*, d.name AS dataset_name
            FROM outline_runs r
            JOIN learning_datasets d ON d.id=r.dataset_id
            WHERE 1=1
        """
        params: list[Any] = []
        if dataset_id:
            query += " AND r.dataset_id=?"
            params.append(dataset_id)
        if not include_deleted:
            query += " AND r.deleted_at IS NULL"
        query += " ORDER BY r.created_at DESC, r.id"
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(query, params).fetchall()
            ]

    def delete_outline_run(self, run_id: str) -> bool:
        with self.connect() as connection:
            item_ids = [
                row["id"]
                for row in connection.execute(
                    "SELECT id FROM outline_items WHERE run_id=?",
                    (run_id,),
                ).fetchall()
            ]
            if item_ids:
                placeholders = ",".join("?" for _ in item_ids)
                connection.execute(
                    f"""
                    DELETE FROM outline_item_versions
                    WHERE item_id IN ({placeholders})
                    """,
                    item_ids,
                )
            connection.execute(
                "DELETE FROM outline_relations WHERE run_id=?",
                (run_id,),
            )
            connection.execute(
                "DELETE FROM outline_items WHERE run_id=?",
                (run_id,),
            )
            cursor = connection.execute(
                "DELETE FROM outline_runs WHERE id=?",
                (run_id,),
            )
            return cursor.rowcount > 0

    def save_outline_items(
        self,
        *,
        run_id: str,
        items: Sequence[OutlineItem],
        change_source: str = "rule",
    ) -> list[str]:
        now = utc_now()
        saved_ids: list[str] = []
        with self.connect() as connection:
            for ordinal, item in enumerate(items, start=1):
                item_id = item.id or f"{run_id}_{item.item_key}"
                saved_ids.append(item_id)
                key_points = json.dumps(item.key_points, ensure_ascii=False)
                indicator_codes = json.dumps(
                    item.indicator_codes,
                    ensure_ascii=False,
                )
                requirements = json.dumps(
                    [entry.model_dump() for entry in item.requirements],
                    ensure_ascii=False,
                )
                sources = json.dumps(
                    [entry.model_dump() for entry in item.sources],
                    ensure_ascii=False,
                )
                warnings = json.dumps(
                    item.review_warnings,
                    ensure_ascii=False,
                )
                connection.execute(
                    """
                    INSERT INTO outline_items
                    (id, run_id, item_key, ordinal, theme_code, theme_name,
                     item_type, content, key_points_json, indicator_codes_json,
                     requirements_json, sources_json, notes,
                     review_warnings_json, status, current_version,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        ordinal=excluded.ordinal,
                        theme_code=excluded.theme_code,
                        theme_name=excluded.theme_name,
                        item_type=excluded.item_type,
                        content=excluded.content,
                        key_points_json=excluded.key_points_json,
                        indicator_codes_json=excluded.indicator_codes_json,
                        requirements_json=excluded.requirements_json,
                        sources_json=excluded.sources_json,
                        notes=excluded.notes,
                        review_warnings_json=excluded.review_warnings_json,
                        status=excluded.status,
                        current_version=outline_items.current_version + 1,
                        updated_at=excluded.updated_at,
                        deleted_at=NULL
                    """,
                    (
                        item_id,
                        run_id,
                        item.item_key,
                        item.ordinal or ordinal,
                        item.theme_code or None,
                        item.theme_name or None,
                        item.item_type,
                        item.content,
                        key_points,
                        indicator_codes,
                        requirements,
                        sources,
                        item.notes or None,
                        warnings,
                        item.status,
                        now,
                        now,
                    ),
                )
                version = connection.execute(
                    """
                    SELECT COUNT(*) FROM outline_item_versions WHERE item_id=?
                    """,
                    (item_id,),
                ).fetchone()[0] + 1
                connection.execute(
                    """
                    INSERT INTO outline_item_versions
                    (item_id, version, content, key_points_json,
                     indicator_codes_json, requirements_json, sources_json,
                     notes, change_source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item_id,
                        version,
                        item.content,
                        key_points,
                        indicator_codes,
                        requirements,
                        sources,
                        item.notes or None,
                        change_source,
                        now,
                    ),
                )
        return saved_ids

    def list_outline_items(
        self,
        run_id: str,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT * FROM outline_items WHERE run_id=?
        """
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        query += " ORDER BY ordinal, item_key"
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(query, (run_id,)).fetchall()
            ]

    def get_outline_item(self, item_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM outline_items WHERE id=?",
                (item_id,),
            ).fetchone()
            return dict(row) if row else None

    def update_outline_item(
        self,
        *,
        item_id: str,
        content: str,
        key_points: Sequence[str],
        indicator_codes: Sequence[str],
        requirements: Sequence[OutlineRequirement],
        sources: Sequence[OutlineEvidence],
        notes: str,
        theme_code: str = "",
        theme_name: str = "",
        item_type: str = "knowledge_point",
        status: str | None = None,
    ) -> None:
        now = utc_now()
        key_points_json = json.dumps(list(key_points), ensure_ascii=False)
        indicator_codes_json = json.dumps(
            list(indicator_codes),
            ensure_ascii=False,
        )
        requirements_json = json.dumps(
            [entry.model_dump() for entry in requirements],
            ensure_ascii=False,
        )
        sources_json = json.dumps(
            [entry.model_dump() for entry in sources],
            ensure_ascii=False,
        )
        with self.connect() as connection:
            current = connection.execute(
                "SELECT * FROM outline_items WHERE id=? AND deleted_at IS NULL",
                (item_id,),
            ).fetchone()
            if current is None:
                raise ValueError("找不到该学习内容")
            version = current["current_version"] + 1
            resolved_status = status or OutlineItemStatus.DRAFT.value
            connection.execute(
                """
                UPDATE outline_items
                SET content=?, key_points_json=?, indicator_codes_json=?,
                    requirements_json=?, sources_json=?, notes=?, theme_code=?,
                    theme_name=?, item_type=?, status=?, current_version=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    content,
                    key_points_json,
                    indicator_codes_json,
                    requirements_json,
                    sources_json,
                    notes or None,
                    theme_code or None,
                    theme_name or None,
                    item_type,
                    resolved_status,
                    version,
                    now,
                    item_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO outline_item_versions
                (item_id, version, content, key_points_json,
                 indicator_codes_json, requirements_json, sources_json,
                 notes, change_source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'manual', ?)
                """,
                (
                    item_id,
                    version,
                    content,
                    key_points_json,
                    indicator_codes_json,
                    requirements_json,
                    sources_json,
                    notes or None,
                    now,
                ),
            )

    def set_outline_items_status(
        self,
        item_ids: Sequence[str],
        status: str,
    ) -> int:
        if status not in set(OutlineItemStatus):
            raise ValueError("无效的梳理状态")
        if not item_ids:
            return 0
        placeholders = ",".join("?" for _ in item_ids)
        with self.connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE outline_items SET status=?, updated_at=?
                WHERE deleted_at IS NULL AND id IN ({placeholders})
                """,
                (status, utc_now(), *item_ids),
            )
            return cursor.rowcount

    def soft_delete_outline_items(self, item_ids: Sequence[str]) -> int:
        if not item_ids:
            return 0
        placeholders = ",".join("?" for _ in item_ids)
        now = utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE outline_items
                SET deleted_at=?, updated_at=?
                WHERE deleted_at IS NULL AND id IN ({placeholders})
                """,
                (now, now, *item_ids),
            )
            return cursor.rowcount

    def save_outline_relations(
        self,
        *,
        run_id: str,
        relations: Sequence[OutlineRelation],
        origin: str = "rule",
    ) -> list[str]:
        now = utc_now()
        saved: list[str] = []
        with self.connect() as connection:
            for relation in relations:
                relation_id = relation.id or f"rel_{uuid.uuid4().hex[:12]}"
                saved.append(relation_id)
                connection.execute(
                    """
                    INSERT INTO outline_relations
                    (id, run_id, source_item_id, target_item_id,
                     relation_type, note, basis, evidence_json, origin,
                     status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        relation_type=excluded.relation_type,
                        note=excluded.note,
                        basis=excluded.basis,
                        evidence_json=excluded.evidence_json,
                        status=excluded.status,
                        deleted_at=NULL
                    """,
                    (
                        relation_id,
                        run_id,
                        relation.source_item_id,
                        relation.target_item_id,
                        relation.relation_type,
                        relation.note,
                        relation.basis,
                        json.dumps(
                            [entry.model_dump() for entry in relation.evidence],
                            ensure_ascii=False,
                        ),
                        origin,
                        relation.status,
                        now,
                    ),
                )
        return saved

    def list_outline_relations(
        self,
        run_id: str,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT r.*, s.item_key AS source_item_key,
                   s.content AS source_content,
                   t.item_key AS target_item_key,
                   t.content AS target_content
            FROM outline_relations r
            JOIN outline_items s ON s.id=r.source_item_id
            JOIN outline_items t ON t.id=r.target_item_id
            WHERE r.run_id=?
        """
        if not include_deleted:
            query += " AND r.deleted_at IS NULL"
        query += " ORDER BY r.created_at, r.id"
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(query, (run_id,)).fetchall()
            ]

    def add_outline_relation(
        self,
        *,
        run_id: str,
        source_item_id: str,
        target_item_id: str,
        relation_type: str,
        note: str,
        basis: str = "",
        evidence: Sequence[OutlineEvidence] | None = None,
        origin: str = "manual",
    ) -> dict[str, Any]:
        if source_item_id == target_item_id:
            raise ValueError("不能把学习内容关联到自身")
        if relation_type not in set(OutlineRelationType):
            raise ValueError("无效的联系类型")
        if not note.strip():
            raise ValueError("请填写联系说明")
        relation_id = f"rel_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        with self.connect() as connection:
            source = connection.execute(
                """
                SELECT id FROM outline_items
                WHERE id=? AND run_id=? AND deleted_at IS NULL
                """,
                (source_item_id, run_id),
            ).fetchone()
            target = connection.execute(
                """
                SELECT id FROM outline_items
                WHERE id=? AND run_id=? AND deleted_at IS NULL
                """,
                (target_item_id, run_id),
            ).fetchone()
            if source is None or target is None:
                raise ValueError("联系两端必须属于本次梳理")
            connection.execute(
                """
                INSERT INTO outline_relations
                (id, run_id, source_item_id, target_item_id, relation_type,
                 note, basis, evidence_json, origin, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?)
                """,
                (
                    relation_id,
                    run_id,
                    source_item_id,
                    target_item_id,
                    relation_type,
                    note.strip(),
                    basis.strip(),
                    json.dumps(
                        [entry.model_dump() for entry in evidence or []],
                        ensure_ascii=False,
                    ),
                    origin,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT r.*, s.item_key AS source_item_key,
                       s.content AS source_content,
                       t.item_key AS target_item_key,
                       t.content AS target_content
                FROM outline_relations r
                JOIN outline_items s ON s.id=r.source_item_id
                JOIN outline_items t ON t.id=r.target_item_id
                WHERE r.id=?
                """,
                (relation_id,),
            ).fetchone()
            return dict(row)

    def delete_outline_relation(self, relation_id: str) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE outline_relations SET deleted_at=?
                WHERE id=? AND deleted_at IS NULL
                """,
                (utc_now(), relation_id),
            )
            if cursor.rowcount == 0:
                raise ValueError("找不到该联系")

    def list_outline_item_versions(self, item_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT * FROM outline_item_versions
                    WHERE item_id=? ORDER BY version DESC
                    """,
                    (item_id,),
                ).fetchall()
            ]
