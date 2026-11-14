from __future__ import annotations

import argparse
from pathlib import Path

from tailorlearn.local_data_migration import LocalDataMigrationError, migrate_local_data
from tailorlearn.paths import AppPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="迁入旧预览环境的本地 TailorLearn 数据")
    default_source = Path(__file__).resolve().parents[2] / "main-preview"
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--target", type=Path, default=AppPaths.default().root)
    args = parser.parse_args()
    try:
        result = migrate_local_data(AppPaths(args.source), AppPaths(args.target))
    except LocalDataMigrationError as exc:
        raise SystemExit(f"迁移未执行：{exc}") from exc
    print(
        "迁移完成："
        f"数据集 {result['datasets']}，学生 {result['students']}，"
        f"材料 {result['materials']}，节点 {result['nodes']}，"
        f"作答 {result['answers']}，对话 {result['chat_messages']}，"
        f"附件 {result['attachment_files']}；备份：{result['backup_path']}"
    )


if __name__ == "__main__":
    main()
