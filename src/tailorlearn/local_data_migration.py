from __future__ import annotations

import os
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tailorlearn.paths import AppPaths


class LocalDataMigrationError(ValueError):
    """Raised when a local data migration cannot be performed safely."""


_CONTENT_TABLES = (
    "material_sets",
    "documents",
    "pages",
    "sections",
    "units",
    "unit_versions",
    "jobs",
    "job_items",
    "learning_datasets",
    "learning_import_batches",
    "learning_field_aliases",
    "learning_import_issues",
    "learning_change_log",
    "analysis_snapshots",
    "ai_diagnosis_jobs",
    "ai_diagnosis_job_items",
    "ai_student_diagnoses",
    "ai_teaching_recommendations",
    "learning_students",
    "learning_requirements",
    "learning_materials",
    "learning_nodes",
    "learning_answers",
    "learning_field_definitions",
    "outline_runs",
    "outline_items",
    "outline_item_versions",
    "outline_relations",
    "teaching_chat_messages",
    "teaching_chat_memory",
)


def _schema_version(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    if row is None:
        raise LocalDataMigrationError("数据库缺少架构版本信息")
    return str(row[0])


def _content_count(connection: sqlite3.Connection) -> int:
    return sum(
        int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in _CONTENT_TABLES
    )


def _source_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "datasets": int(
            connection.execute("SELECT COUNT(*) FROM learning_datasets").fetchone()[0]
        ),
        "students": int(
            connection.execute("SELECT COUNT(*) FROM learning_students").fetchone()[0]
        ),
        "materials": int(
            connection.execute("SELECT COUNT(*) FROM learning_materials").fetchone()[0]
        ),
        "nodes": int(
            connection.execute("SELECT COUNT(*) FROM learning_nodes").fetchone()[0]
        ),
        "answers": int(
            connection.execute("SELECT COUNT(*) FROM learning_answers").fetchone()[0]
        ),
        "chat_messages": int(
            connection.execute("SELECT COUNT(*) FROM teaching_chat_messages").fetchone()[0]
        ),
    }


def _backup_database(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(source)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def migrate_local_data(source: AppPaths, target: AppPaths) -> dict[str, Any]:
    """Safely replace a bootstrap database with a compatible local data snapshot."""
    source.ensure()
    target.ensure()
    if not source.database_file.exists():
        raise LocalDataMigrationError("未找到迁移源数据库")
    if not target.database_file.exists():
        raise LocalDataMigrationError("未找到目标数据库")

    source_connection = sqlite3.connect(source.database_file)
    target_connection = sqlite3.connect(target.database_file)
    try:
        source_version = _schema_version(source_connection)
        target_version = _schema_version(target_connection)
        if source_version != target_version:
            raise LocalDataMigrationError(
                f"数据库架构版本不一致：源 {source_version}，目标 {target_version}"
            )
        if _content_count(target_connection):
            raise LocalDataMigrationError("目标数据库已包含数据，已停止迁移以避免覆盖")
        counts = _source_counts(source_connection)
    finally:
        target_connection.close()
        source_connection.close()

    target_material_files = list(target.materials_dir.rglob("*"))
    if any(path.is_file() for path in target_material_files):
        raise LocalDataMigrationError("目标材料目录已包含文件，已停止迁移以避免覆盖")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = target.db_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"tailorlearn-before-migration-{timestamp}.sqlite3"
    staged_database = target.db_dir / f".migration-{timestamp}.sqlite3"
    staged_materials = target.db_dir / f".materials-migration-{timestamp}"
    original_materials = target.db_dir / f".materials-before-migration-{timestamp}"
    _backup_database(target.database_file, backup_path)
    _backup_database(source.database_file, staged_database)
    try:
        if source.materials_dir.exists():
            shutil.copytree(source.materials_dir, staged_materials)
        else:
            staged_materials.mkdir()
        attachment_files = sum(
            1 for path in staged_materials.rglob("*") if path.is_file()
        )
        os.replace(target.materials_dir, original_materials)
        os.replace(staged_materials, target.materials_dir)
        os.replace(staged_database, target.database_file)
    except Exception:
        if original_materials.exists():
            if target.materials_dir.exists():
                shutil.rmtree(target.materials_dir)
            os.replace(original_materials, target.materials_dir)
        if staged_database.exists():
            staged_database.unlink()
        if staged_materials.exists():
            shutil.rmtree(staged_materials)
        raise
    shutil.rmtree(original_materials)
    counts.update(
        {
            "attachment_files": attachment_files,
            "backup_path": backup_path,
        }
    )
    return counts
