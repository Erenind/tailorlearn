from __future__ import annotations

import json

from tailorlearn.db import Repository
from tailorlearn.models import AIUnit, ExtractedSection, PageResult, SourceRef


def test_repository_units_status_versions_and_soft_delete(tmp_path):
    repo = Repository(tmp_path / "catalog.sqlite3")
    repo.initialize()
    material = repo.create_material_set(
        name="测试课本",
        material_type="textbook",
        subject="数学",
    )
    document, inserted = repo.add_document(
        document_id="doc_test",
        material_set_id=material["id"],
        file_name="slice.pdf",
        sha256="abc123",
        stored_path="materials/test/source/original.pdf",
        order_index=1,
        page_count=1,
    )
    assert inserted
    repo.update_material_metadata(
        material["id"],
        source="inferred:content",
        grade="七年级",
    )
    refreshed = repo.get_material_set(material["id"])
    assert json.loads(refreshed["metadata_sources_json"])["grade"] == "inferred:content"
    repo.replace_pages(
        document["id"],
        [
            PageResult(
                page_number=1,
                raw_text="不等式的性质",
                final_text="不等式的性质",
                source_kind="text",
            )
        ],
    )
    section = ExtractedSection(
        id="doc_test_p001_s001",
        document_id="doc_test",
        path="第一章 / 1.1 不等式",
        title="1.1 不等式",
        level=2,
        order_index=1,
        page_start=1,
        page_end=1,
        content="【第 1 页】\n不等式的性质",
    )
    repo.replace_sections("doc_test", [section])
    unit = AIUnit(
        unit_type="property",
        title="不等式的性质",
        content="不等式两边加同一个数，不等号方向不变。",
        source_refs=[SourceRef(page=1, quote="不等式的性质")],
    )
    ids = repo.save_units_for_section(
        document_id="doc_test",
        section_id=section.id,
        units=[unit],
        ai_profile="Vendor",
        ai_base_url="https://example.test/v1",
        ai_model="model",
        raw_response_path="runs/test.raw.txt",
    )
    assert ids == ["doc_test_p001_s001_u001"]
    saved = repo.get_unit(ids[0])
    assert saved is not None
    assert json.loads(saved["source_refs_json"])[0]["page"] == 1
    assert saved["current_version"] == 1
    assert saved["status"] == "draft"

    repo.update_unit(
        unit_id=ids[0],
        title="不等式的性质（修订）",
        unit_type="property",
        content="修订内容",
    )
    saved = repo.get_unit(ids[0])
    assert saved is not None
    assert saved["current_version"] == 2
    assert saved["status"] == "draft"

    assert repo.set_units_status(ids, "verified") == 1
    assert repo.get_unit(ids[0])["status"] == "verified"
    assert repo.soft_delete_units(ids) == 1
    assert repo.list_units(document_id="doc_test") == []
    deleted = repo.list_units(document_id="doc_test", include_deleted=True)
    assert deleted[0]["deleted_at"]
    assert repo.restore_units(ids) == 1
    assert len(repo.list_units(document_id="doc_test")) == 1
