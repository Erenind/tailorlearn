from __future__ import annotations

from pathlib import Path

from tailorlearn.ai import OpenAICompatibleClient
from tailorlearn.db import Repository
from tailorlearn.ingest import IngestionService
from tailorlearn.models import ProviderProfile
from tailorlearn.paths import AppPaths


def test_structure_document_persists_units_and_raw_response(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "res" / "B02_相关资料切片.pdf"
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    service = IngestionService(paths, repository)
    _, documents = service.create_material_from_paths(
        name="七年级数学课本切片",
        material_type="textbook",
        pdf_paths=[source],
    )
    service.process_document(documents[0]["id"])

    def fake_complete(self, messages):
        return (
            '{"units":[{"unit_type":"concept","title":"测试单元",'
            '"content":"忠实整理内容","source_refs":[],'
            '"related_unit_ids":[],"review_warnings":[]}]} ',
            "stop",
        )

    monkeypatch.setattr(OpenAICompatibleClient, "_complete", fake_complete)
    profile = ProviderProfile(
        id="provider_test",
        vendor_name="Test",
        base_url="https://example.test/v1",
        model="test-model",
        api_key="test-key",
    )
    result = service.structure_document(
        document_id=documents[0]["id"],
        profile=profile,
    )
    assert result["failed"] == 0
    units = repository.list_units(document_id=documents[0]["id"])
    assert units
    assert all(unit["status"] == "needs_review" for unit in units)
    # The run is nested under the material/document; verify at least one raw file.
    assert list(
        (paths.db_dir / "materials").glob(
            f"*/{documents[0]['id']}/runs/{result['run_id']}/chunks/*.raw.txt"
        )
    )
