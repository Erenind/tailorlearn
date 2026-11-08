from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tailorlearn.db import Repository
from tailorlearn.ingest import IngestionService
from tailorlearn.paths import AppPaths
from tailorlearn.providers import ProviderStore


def _services() -> tuple[AppPaths, Repository, IngestionService, ProviderStore]:
    paths = AppPaths.default()
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    return paths, repository, IngestionService(paths, repository), ProviderStore(
        paths.provider_file
    )


def _run_ui() -> int:
    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    return int(streamlit_cli.main() or 0)


def _list_materials() -> int:
    _, repository, _, _ = _services()
    materials = repository.list_material_sets()
    if not materials:
        print("没有材料")
        return 0
    for material in materials:
        print(
            f"{material['id']}  {material['material_type']}  "
            f"{material['name']}  v{material['version']}"
        )
        for document in repository.list_documents(material["id"]):
            print(
                f"  {document['id']}  {document['file_name']}  "
                f"{document['extraction_status']}  {document['page_count']} pages"
            )
    return 0


def _import_pdfs(args: argparse.Namespace) -> int:
    _, repository, ingestion, _ = _services()
    paths = [Path(value).expanduser().resolve() for value in args.pdfs]
    material, documents = ingestion.create_material_from_paths(
        name=args.name,
        material_type=args.material_type,
        pdf_paths=paths,
        subject=args.subject,
        stage=args.stage,
        grade=args.grade,
        semester=args.semester,
        edition=args.edition,
        append_to=args.append_to,
    )
    for document in documents:
        def progress(current: int, total: int, message: str) -> None:
            if current == 1 or current == total or current % 10 == 0:
                print(f"{document['file_name']}: {message}", flush=True)

        result = ingestion.process_document(document["id"], progress)
        print(
            f"{document['file_name']}: {result['pages']} 页，"
            f"{result['sections']} 个章节块"
        )
    print(f"材料：{material['id']}  {material['name']}")
    return 0


def _structure(args: argparse.Namespace) -> int:
    _, repository, ingestion, providers = _services()
    profile = providers.active()
    if profile is None:
        print("请先在 AI 参数设置页保存并启用一个厂商配置")
        return 2
    if args.document_id:
        documents = [repository.get_document(args.document_id)]
    else:
        documents = [
            document
            for material in repository.list_material_sets()
            for document in repository.list_documents(material["id"])
        ]
    documents = [
        document
        for document in documents
        if document and document["extraction_status"] == "succeeded"
    ]
    if not documents:
        print("没有可结构化的已提取文档")
        return 1
    has_failure = False
    for document in documents:
        print(f"结构化：{document['file_name']}", flush=True)
        result = ingestion.structure_document(
            document_id=document["id"],
            profile=profile,
            force=args.force,
            progress=lambda current, total, title: print(
                f"  {current}/{total} {title}",
                flush=True,
            ),
        )
        print(
            f"  完成 {result['completed']}，失败 {result['failed']}，"
            f"单元 {result['units']}"
        )
        for error in result["errors"]:
            print(f"  ERROR {error}")
        has_failure = has_failure or bool(result["failed"])
    return 1 if has_failure else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tailorlearn")
    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="启动 Streamlit 界面")
    ui_parser.set_defaults(handler=_run_ui)

    list_parser = subparsers.add_parser("list", help="列出材料和文档")
    list_parser.set_defaults(handler=lambda args: _list_materials())

    import_parser = subparsers.add_parser(
        "import-pdf",
        help="导入一个版本的一套材料并完成 OCR、切分",
    )
    import_parser.add_argument("--name", required=True)
    import_parser.add_argument(
        "--material-type",
        required=True,
        choices=["textbook", "curriculum_standard"],
    )
    import_parser.add_argument("--subject")
    import_parser.add_argument("--stage")
    import_parser.add_argument("--grade")
    import_parser.add_argument("--semester")
    import_parser.add_argument("--edition")
    import_parser.add_argument("--append-to")
    import_parser.add_argument("pdfs", nargs="+")
    import_parser.set_defaults(handler=_import_pdfs)

    structure_parser = subparsers.add_parser(
        "structure",
        help="使用当前厂商配置结构化已提取文档",
    )
    structure_parser.add_argument("--document-id")
    structure_parser.add_argument("--force", action="store_true")
    structure_parser.set_defaults(handler=_structure)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        args.command = "ui"
        args.handler = lambda _: _run_ui()
    raise SystemExit(args.handler(args))

