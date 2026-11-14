from __future__ import annotations

import os
from pathlib import Path

import pytest

from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.local_data_migration import LocalDataMigrationError, migrate_local_data
from tailorlearn.paths import AppPaths


def _create_source(root: Path) -> AppPaths:
    paths = AppPaths(root)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    learning = LearningDataService(repository, paths.db_dir)
    dataset = learning.create_dataset("迁移源")
    learning.create_record(
        dataset["id"],
        "student",
        {"student_code": "D01", "class_code": "DC01", "grade": 7},
    )
    attachment = paths.materials_dir / "sample.txt"
    attachment.write_text("迁移附件", encoding="utf-8")
    return paths


def _create_target(root: Path) -> AppPaths:
    paths = AppPaths(root)
    paths.ensure()
    Repository(paths.database_file).initialize()
    return paths


def test_local_data_migration_copies_database_and_materials(tmp_path: Path) -> None:
    source = _create_source(tmp_path / "source")
    target = _create_target(tmp_path / "target")

    result = migrate_local_data(source, target)

    learning = LearningDataService(Repository(target.database_file), target.db_dir)
    datasets = learning.list_datasets()
    assert result["datasets"] == 1
    assert result["backup_path"].exists()
    assert [row["name"] for row in datasets] == ["迁移源"]
    assert learning.list_records(datasets[0]["id"], "student")[0]["student_code"] == "D01"
    assert (target.materials_dir / "sample.txt").read_text(encoding="utf-8") == "迁移附件"


def test_local_data_migration_refuses_nonempty_target(tmp_path: Path) -> None:
    source = _create_source(tmp_path / "source")
    target = _create_target(tmp_path / "target")
    learning = LearningDataService(Repository(target.database_file), target.db_dir)
    learning.create_dataset("已有数据")

    with pytest.raises(LocalDataMigrationError, match="目标数据库已包含数据"):
        migrate_local_data(source, target)


def test_local_data_migration_restores_empty_materials_when_database_swap_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _create_source(tmp_path / "source")
    target = _create_target(tmp_path / "target")
    original_database = target.database_file.read_bytes()
    real_replace = os.replace

    def fail_database_swap(source_path: str | Path, target_path: str | Path) -> None:
        if Path(target_path) == target.database_file:
            raise OSError("模拟数据库替换失败")
        real_replace(source_path, target_path)

    monkeypatch.setattr("tailorlearn.local_data_migration.os.replace", fail_database_swap)

    with pytest.raises(OSError, match="模拟数据库替换失败"):
        migrate_local_data(source, target)

    assert target.database_file.read_bytes() == original_database
    assert list(target.materials_dir.rglob("*")) == []
