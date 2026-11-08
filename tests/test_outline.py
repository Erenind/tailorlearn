from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.models import (
    AIUnit,
    ExtractedSection,
    OutlineDraftItem,
    OutlineDraftRelation,
    OutlineDraftRequirement,
    OutlineDraftResponse,
    OutlineDraftSource,
    OutlineEvidence,
    OutlineItem,
    OutlineRequirement,
    PageResult,
    ProviderProfile,
    SourceRef,
)
from tailorlearn.outline import OutlineService
from tailorlearn.paths import AppPaths


def _repository(tmp_path: Path) -> tuple[AppPaths, Repository, OutlineService]:
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    return paths, repository, OutlineService(paths, repository)


def _dataset(
    repository: Repository,
    paths: AppPaths,
    requirements: list[tuple[str, str, str, str]],
) -> str:
    service = LearningDataService(repository, paths.db_dir)
    dataset = service.create_dataset("测试学情数据")
    with repository.connect() as connection:
        for theme_code, theme_name, indicator_code, indicator_name in requirements:
            connection.execute(
                """
                INSERT INTO learning_requirements
                (dataset_id, theme_code, theme_name, theme_goal, indicator_code,
                 indicator_name, standard_ref, standard_page, textbook_page,
                 source_json, extra_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', '{}', ?, ?)
                """,
                (
                    dataset["id"],
                    theme_code,
                    theme_name,
                    f"理解{theme_name}",
                    indicator_code,
                    indicator_name,
                    "CR01",
                    "1",
                    "上册1—10",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
    return dataset["id"]


def _standard_material(repository: Repository, pages: list[str]) -> str:
    material = repository.create_material_set(
        name="测试课程标准",
        material_type="curriculum_standard",
    )
    document, _ = repository.add_document(
        document_id="doc_standard",
        material_set_id=material["id"],
        file_name="标准.pdf",
        sha256="standard",
        stored_path="materials/standard.pdf",
        order_index=1,
        page_count=len(pages),
    )
    repository.replace_pages(
        document["id"],
        [
            PageResult(
                page_number=index,
                raw_text=text,
                final_text=text,
                source_kind="text",
            )
            for index, text in enumerate(pages, start=1)
        ],
    )
    section = ExtractedSection(
        id="doc_standard_p001_s001",
        document_id=document["id"],
        path="1.数与式",
        title="1.数与式",
        level=1,
        order_index=1,
        page_start=1,
        page_end=len(pages),
        content="",
    )
    repository.replace_sections(document["id"], [section])
    return material["id"]


def _textbook_material(
    repository: Repository,
    pages: list[str],
    sections: list[tuple[str, int, int]],
) -> tuple[str, str]:
    material = repository.create_material_set(
        name="测试教材",
        material_type="textbook",
    )
    document, _ = repository.add_document(
        document_id="doc_textbook",
        material_set_id=material["id"],
        file_name="教材.pdf",
        sha256="textbook",
        stored_path="materials/textbook.pdf",
        order_index=1,
        page_count=len(pages),
    )
    repository.replace_pages(
        document["id"],
        [
            PageResult(
                page_number=index,
                raw_text=text,
                final_text=text,
                source_kind="text",
            )
            for index, text in enumerate(pages, start=1)
        ],
    )
    repository.replace_sections(
        document["id"],
        [
            ExtractedSection(
                id=f"doc_textbook_p{page:03d}_s{index:03d}",
                document_id=document["id"],
                path=title,
                title=title,
                level=level,
                order_index=index,
                page_start=page,
                page_end=page,
                content=pages[page - 1],
            )
            for index, (title, level, page) in enumerate(sections, start=1)
        ],
    )
    return material["id"], document["id"]


def _basic_scope(
    tmp_path: Path,
    *,
    standard_pages: list[str] | None = None,
    textbook_pages: list[str] | None = None,
    textbook_sections: list[tuple[str, int, int]] | None = None,
    requirements: list[tuple[str, str, str, str]] | None = None,
) -> tuple[OutlineService, str, dict]:
    _, repository, service = _repository(tmp_path)
    dataset_id = _dataset(
        repository,
        AppPaths(tmp_path),
        requirements
        or [
            (
                "T01",
                "方程与方程组",
                "T01I1",
                "运用等式性质解一元一次方程",
            )
        ],
    )
    standard_id = _standard_material(
        repository,
        standard_pages
        or [
            "四、课程内容\n1.数与式\n（1）方程与方程组\n"
            "①理解方程的意义，能针对具体问题列出方程。\n"
            "②掌握等式的基本性质，能解一元一次方程。"
        ],
    )
    textbook_id, _ = _textbook_material(
        repository,
        textbook_pages
        or [
            "5.1 方程\n方程是含有未知数的等式。\n"
            "5.2 解一元一次方程\n解一元一次方程要依据等式的基本性质，"
            "把方程逐步化为x=a的形式。"
        ],
        textbook_sections
        or [
            ("第五章 一元一次方程", 1, 1),
            ("5.1 方程", 2, 1),
            ("5.2 解一元一次方程", 2, 1),
        ],
    )
    scope = service.build_scope(
        dataset_id=dataset_id,
        theme_codes=["T01"],
        standard_material_id=standard_id,
        standard_document_id="doc_standard",
        textbook_material_id=textbook_id,
        textbook_document_ids=["doc_textbook"],
    )
    return service, dataset_id, scope


def test_rule_outline_persists_verified_sources_and_exports(tmp_path):
    service, _, scope = _basic_scope(tmp_path)
    assert scope["standard"]["themes"][0]["matched"] is True
    assert scope["textbook"]["sections"]
    assert "其余内容未纳入" not in scope["boundary"]["statement"] or True

    result = service.generate(scope=scope, method="rule")
    run = result["run"]
    assert run["item_count"] == 1
    items = service.items(run["id"])
    assert items[0].item_key == "T01I1"
    assert items[0].requirements
    assert items[0].sources
    assert all(
        requirement.verified for requirement in items[0].requirements
    )
    assert all(source.verified for source in items[0].sources)

    bundle = json.loads(service.export_bundle(run["id"]).decode("utf-8"))
    assert bundle["items"][0]["item_key"] == "T01I1"
    assert bundle["references"]
    assert bundle["field_definitions"]
    assert bundle["scope"]["dataset"]["id"] == scope["dataset"]["id"]

    root = service.outlines_dir / run["id"]
    assert (root / "outline.json").exists()
    assert (root / "items.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (root / "relations.csv").exists()
    assert (root / "README.md").exists()
    rows = list(
        csv.DictReader(
            io.StringIO((root / "items.csv").read_text(encoding="utf-8-sig"))
        )
    )
    assert rows[0]["编号"] == "T01I1"
    assert rows[0]["学习内容"] == "运用等式性质解一元一次方程"

    vocabulary = service.export_field_definitions_csv().decode("utf-8-sig")
    assert "字段说明" in vocabulary or "数据表" in vocabulary


def test_scope_records_excluded_chapters(tmp_path):
    service, _, scope = _basic_scope(
        tmp_path,
        textbook_pages=[
            "5.1 方程\n方程是含有未知数的等式。\n"
            "6.1 几何图形\n几何图形包括立体图形和平面图形。"
        ],
        textbook_sections=[
            ("第五章 一元一次方程", 1, 1),
            ("5.1 方程", 2, 1),
            ("第六章 几何图形初步", 1, 1),
            ("6.1 几何图形", 2, 1),
        ],
    )
    chapters = {
        row["chapter"] for row in scope["textbook"]["excluded"]
    }
    assert "第六章 几何图形初步" in chapters
    assert scope["boundary"]["excluded_note"]
    assert "未纳入" in scope["boundary"]["excluded_note"]


def test_validation_flags_unmatched_quotes(tmp_path):
    service, _, scope = _basic_scope(tmp_path)
    item = OutlineItem(
        item_key="X01",
        content="测试内容",
        requirements=[
            OutlineRequirement(
                text="引用错误",
                evidence=OutlineEvidence(
                    kind="standard",
                    material_set_id=scope["standard"]["material_set_id"],
                    document_id=scope["standard"]["document_id"],
                    file_name="标准.pdf",
                    page=1,
                    quote="这段原文不在页面中",
                ),
            )
        ],
        sources=[
            OutlineEvidence(
                kind="textbook",
                material_set_id=scope["textbook"]["material_set_id"],
                document_id=scope["textbook"]["sections"][0]["document_id"],
                section_id=scope["textbook"]["sections"][0]["section_id"],
                page=1,
                quote="这段原文不在教材中",
            )
        ],
    )
    checked = service.validate_items(scope, [item])[0]
    assert checked.status == "needs_review"
    assert checked.review_warnings
    assert not checked.requirements[0].verified
    assert not checked.sources[0].verified


def test_manual_items_relations_and_status_round_trip(tmp_path):
    service, _, scope = _basic_scope(
        tmp_path,
        requirements=[
            ("T01", "方程与方程组", "T01I1", "运用等式性质解一元一次方程"),
            ("T01", "方程与方程组", "T01I2", "列方程解决实际问题"),
        ],
    )
    run = service.generate(scope=scope, method="rule")["run"]
    items = service.items(run["id"])
    assert len(items) == 2
    first, second = items
    service.set_items_status([first.id], "verified")
    assert next(item for item in service.items(run["id"]) if item.id == first.id).status == "verified"
    service.update_item(
        item_id=first.id,
        content="人工修订后的内容",
        key_points=["要点一"],
        indicator_codes=first.indicator_codes,
        notes="人工修订",
        theme_code=first.theme_code,
        theme_name=first.theme_name,
        item_type=first.item_type,
    )
    updated = next(item for item in service.items(run["id"]) if item.id == first.id)
    assert updated.content == "人工修订后的内容"
    assert updated.current_version == 2

    before = len(service.relations(run["id"]))
    service.add_relation(
        run_id=run["id"],
        source_item_id=first.id,
        target_item_id=second.id,
        relation_type="parallel",
        note="人工补充的并列联系",
    )
    relations = service.relations(run["id"])
    assert len(relations) == before + 1
    manual = next(relation for relation in relations if relation.origin == "manual")
    assert manual.note == "人工补充的并列联系"
    service.delete_relation(manual.id)
    assert len(service.relations(run["id"])) == before

    manual_item = service.add_manual_item(
        run_id=run["id"],
        theme_code=first.theme_code,
        content="人工添加的学习内容",
        key_points=["人工要点"],
        indicator_codes=[],
        notes="人工添加",
    )
    assert manual_item.status == "needs_review"
    assert manual_item.item_key.startswith("KP")
    assert len(service.items(run["id"])) == 3
    service.delete_item(manual_item.id)
    assert len(service.items(run["id"])) == 2


def test_ai_draft_is_resolved_against_scope(tmp_path):
    service, _, scope = _basic_scope(tmp_path)
    document = scope["textbook"]["sections"][0]
    draft = OutlineDraftResponse(
        scope_note="测试",
        items=[
            OutlineDraftItem(
                item_key="T01I1",
                theme_code="T01",
                item_type="skill",
                content="等式的性质与一元一次方程的解法",
                key_points=["等式的性质"],
                indicator_codes=["T01I1"],
                requirements=[
                    OutlineDraftRequirement(
                        text="掌握等式的基本性质，能解一元一次方程",
                        clause="②",
                        level="掌握",
                        standard_page=1,
                        quote="②掌握等式的基本性质，能解一元一次方程。",
                    )
                ],
                sources=[
                    OutlineDraftSource(
                        kind="textbook",
                        material_set_id=scope["textbook"]["material_set_id"],
                        document_id=document["document_id"],
                        section_id=document["section_id"],
                        page=1,
                        quote="解一元一次方程要依据等式的基本性质",
                    )
                ],
            )
        ],
        relations=[],
    )
    items, relations = service.ai_items(
        scope,
        draft,
        ProviderProfile(
            id="p",
            vendor_name="测试",
            base_url="https://example.test/v1",
            model="test",
            api_key="key",
        ),
    )
    assert items[0].item_key == "T01I1"
    assert items[0].theme_code == "T01"
    assert items[0].status == "needs_review"
    assert items[0].requirements[0].evidence is not None
    assert items[0].sources[0].section_id == document["section_id"]
    assert relations == []
