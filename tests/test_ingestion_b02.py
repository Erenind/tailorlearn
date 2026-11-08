from __future__ import annotations

import json
from pathlib import Path

from tailorlearn.db import Repository
from tailorlearn.ingest import IngestionService
from tailorlearn.models import PageResult
from tailorlearn.paths import AppPaths


def test_b02_text_extraction_and_split_creates_stable_records(tmp_path):
    source = Path(__file__).resolve().parents[1] / "res" / "B02_相关资料切片.pdf"
    paths = AppPaths(tmp_path)
    paths.ensure()
    repo = Repository(paths.database_file)
    repo.initialize()
    service = IngestionService(paths, repo)
    material, documents = service.create_material_from_paths(
        name="七年级数学课本切片",
        material_type="textbook",
        pdf_paths=[source],
    )
    result = service.process_document(documents[0]["id"])

    assert result["pages"] == 26
    assert result["sections"] >= 5
    refreshed_material = repo.get_material_set(material["id"])
    assert refreshed_material["subject"] == "数学"
    assert (
        json.loads(refreshed_material["metadata_sources_json"])["subject"]
        == "inferred:content"
    )
    assert (paths.db_dir / documents[0]["stored_path"]).exists()
    sections = repo.list_sections(documents[0]["id"])
    assert sections[0]["id"].startswith("doc_")
    assert any("11.1 不等式" == item["title"] for item in sections)
    pages = repo.list_pages(documents[0]["id"])
    assert all(page["source_kind"] == "text" for page in pages)


def test_process_document_resumes_from_cached_pages(tmp_path):
    source = Path(__file__).resolve().parents[1] / "res" / "B02_相关资料切片.pdf"
    paths = AppPaths(tmp_path)
    paths.ensure()
    repo = Repository(paths.database_file)
    repo.initialize()
    service = IngestionService(paths, repo)
    _, documents = service.create_material_from_paths(
        name="七年级数学课本切片",
        material_type="textbook",
        pdf_paths=[source],
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
    assert result["pages"] == 26
    assert len(pages) == 26
    assert pages[0]["final_text"] == "缓存页"
    assert pages[0]["edited"] == 0
