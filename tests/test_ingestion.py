from __future__ import annotations

from pathlib import Path

from tailorlearn.db import Repository
from tailorlearn.ingest import IngestionService
from tailorlearn.models import PageResult
from tailorlearn.paths import AppPaths


def test_text_extraction_and_split_creates_stable_records(
    tmp_path: Path,
    sample_pdf: Path,
):
    paths = AppPaths(tmp_path)
    paths.ensure()
    repo = Repository(paths.database_file)
    repo.initialize()
    service = IngestionService(paths, repo)
    _, documents = service.create_material_from_paths(
        name="测试课本切片",
        material_type="textbook",
        pdf_paths=[sample_pdf],
    )
    result = service.process_document(documents[0]["id"])

    assert result["pages"] == 3
    assert result["sections"] == 3
    assert (paths.db_dir / documents[0]["stored_path"]).exists()
    sections = repo.list_sections(documents[0]["id"])
    assert sections[0]["id"].startswith("doc_")
    pages = repo.list_pages(documents[0]["id"])
    assert all(page["source_kind"] == "text" for page in pages)


def test_process_document_resumes_from_cached_pages(
    tmp_path: Path,
    sample_pdf: Path,
):
    paths = AppPaths(tmp_path)
    paths.ensure()
    repo = Repository(paths.database_file)
    repo.initialize()
    service = IngestionService(paths, repo)
    _, documents = service.create_material_from_paths(
        name="测试课本切片",
        material_type="textbook",
        pdf_paths=[sample_pdf],
    )
    document_id = documents[0]["id"]
    repo.upsert_page(
        document_id,
        PageResult(
            page_number=1,
            raw_text="缓存页",
            final_text="缓存页",
            source_kind="edited",
        ),
    )
    result = service.process_document(document_id)
    pages = repo.list_pages(document_id)
    assert result["pages"] == 3
    assert len(pages) == 3
    assert pages[0]["final_text"] == "缓存页"


def test_reprocess_document_forces_fresh_text_extraction(
    tmp_path: Path,
    sample_pdf: Path,
):
    paths = AppPaths(tmp_path)
    paths.ensure()
    repo = Repository(paths.database_file)
    repo.initialize()
    service = IngestionService(paths, repo)
    _, documents = service.create_material_from_paths(
        name="重新识别测试课本",
        material_type="textbook",
        pdf_paths=[sample_pdf],
    )
    document_id = documents[0]["id"]
    service.process_document(document_id)
    repo.update_page_text(document_id, 1, "人工修改后的缓存文本")

    result = service.reprocess_document(document_id)

    assert result["pages"] == 3
    assert result["structure_mode"] == "rebuilt-sections"
    assert repo.list_pages(document_id)[0]["final_text"] != "人工修改后的缓存文本"


def test_reimport_restores_soft_deleted_document(
    tmp_path: Path,
    sample_pdf: Path,
):
    paths = AppPaths(tmp_path)
    paths.ensure()
    repo = Repository(paths.database_file)
    repo.initialize()
    service = IngestionService(paths, repo)
    material, documents = service.create_material_from_paths(
        name="重新导入测试课本",
        material_type="textbook",
        pdf_paths=[sample_pdf],
    )
    document_id = documents[0]["id"]
    service.process_document(document_id)
    assert repo.soft_delete_document(document_id) is True
    assert repo.list_documents(material["id"]) == []

    _, imported = service.create_material_from_paths(
        name=material["name"],
        material_type="textbook",
        pdf_paths=[sample_pdf],
        append_to=material["id"],
    )

    assert [document["id"] for document in imported] == [document_id]
    active_documents = repo.list_documents(material["id"])
    assert [document["id"] for document in active_documents] == [document_id]
    assert active_documents[0]["deleted_at"] is None
    assert service.process_document(document_id)["pages"] == 3
