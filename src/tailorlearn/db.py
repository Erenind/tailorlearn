from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from tailorlearn.models import AIUnit, PageResult, UnitStatus


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
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", "3"),
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
                return dict(existing), False
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
