from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pymupdf

from tailorlearn.ai import OpenAICompatibleClient
from tailorlearn.db import Repository
from tailorlearn.extraction import extract_pdf
from tailorlearn.models import (
    ExtractedSection,
    MaterialType,
    PageResult,
    ProviderProfile,
)
from tailorlearn.paths import AppPaths
from tailorlearn.splitting import needs_ai_heading_fallback, split_pages


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _infer_metadata(
    *,
    material_type: str,
    name: str,
    file_names: Sequence[str],
    page_texts: Sequence[str],
) -> dict[str, str]:
    sample = "\n".join([name, *file_names, *page_texts])
    metadata: dict[str, str] = {}
    subject_words = {
        "数学": "数学",
        "语文": "语文",
        "英语": "英语",
        "物理": "物理",
        "化学": "化学",
        "生物学": "生物学",
        "历史": "历史",
        "地理": "地理",
    }
    for needle, value in subject_words.items():
        if needle in sample:
            metadata["subject"] = value
            break
    if material_type == MaterialType.TEXTBOOK.value:
        grade_match = re.search(r"([一二三四五六七八九])年级", sample)
        if grade_match:
            metadata["grade"] = f"{grade_match.group(1)}年级"
    if (
        material_type == MaterialType.STANDARD.value
        and "义务教育" in sample
    ) or ("小学" in sample and "初中" in sample):
        metadata["stage"] = "义务教育"
    elif "初中" in sample:
        metadata["stage"] = "初中"
    elif "小学" in sample:
        metadata["stage"] = "小学"
    elif "高中" in sample:
        metadata["stage"] = "高中"
    for edition in ("人教版", "北师大版", "沪科版", "苏教版", "湘教版"):
        if edition in sample:
            metadata["edition"] = edition
            break
    semester = "上册" if "上册" in sample else "下册" if "下册" in sample else None
    if semester:
        metadata["semester"] = semester
    return metadata


class IngestionService:
    def __init__(self, paths: AppPaths, repository: Repository):
        self.paths = paths
        self.repo = repository

    def add_pdf_path(
        self,
        *,
        source_path: Path,
        material_set_id: str,
        file_name: str,
        order_index: int,
    ) -> tuple[dict[str, Any], bool]:
        source_path = Path(source_path)
        sha = sha256_file(source_path)
        document_id = f"doc_{sha[:8]}_{material_set_id[-6:]}"
        with pymupdf.open(source_path) as document:
            page_count = document.page_count
        relative_path = (
            Path("materials")
            / material_set_id
            / document_id
            / "source"
            / "original.pdf"
        )
        destination = self.paths.db_dir / relative_path
        row, inserted = self.repo.add_document(
            document_id=document_id,
            material_set_id=material_set_id,
            file_name=file_name,
            sha256=sha,
            stored_path=str(relative_path),
            order_index=order_index,
            page_count=page_count,
            extraction_method="text-first-ocr-fallback",
        )
        if inserted or not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != destination.resolve():
                shutil.copy2(source_path, destination)
        self._write_manifest(material_set_id)
        return row, inserted

    def add_pdf_bytes(
        self,
        *,
        content: bytes,
        material_set_id: str,
        file_name: str,
        order_index: int,
    ) -> tuple[dict[str, Any], bool]:
        temp_dir = self.paths.db_dir / ".upload_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{uuid.uuid4().hex}.pdf"
        try:
            temp_path.write_bytes(content)
            return self.add_pdf_path(
                source_path=temp_path,
                material_set_id=material_set_id,
                file_name=file_name,
                order_index=order_index,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def create_material_from_paths(
        self,
        *,
        name: str,
        material_type: str,
        pdf_paths: Sequence[Path],
        subject: str | None = None,
        stage: str | None = None,
        grade: str | None = None,
        semester: str | None = None,
        edition: str | None = None,
        append_to: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not pdf_paths:
            raise ValueError("至少需要一个 PDF 文件")
        material = self.repo.create_material_set(
            name=name,
            material_type=material_type,
            subject=subject,
            stage=stage,
            grade=grade,
            semester=semester,
            edition=edition,
            append_to=append_to,
        )
        documents: list[dict[str, Any]] = []
        for index, path in enumerate(pdf_paths, start=1):
            row, _ = self.add_pdf_path(
                source_path=path,
                material_set_id=material["id"],
                file_name=path.name,
                order_index=index,
            )
            documents.append(row)
        return material, documents

    def create_material_from_bytes(
        self,
        *,
        name: str,
        material_type: str,
        files: Sequence[tuple[str, bytes]],
        subject: str | None = None,
        stage: str | None = None,
        grade: str | None = None,
        semester: str | None = None,
        edition: str | None = None,
        append_to: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not files:
            raise ValueError("至少需要一个 PDF 文件")
        material = self.repo.create_material_set(
            name=name,
            material_type=material_type,
            subject=subject,
            stage=stage,
            grade=grade,
            semester=semester,
            edition=edition,
            append_to=append_to,
        )
        documents: list[dict[str, Any]] = []
        for index, (file_name, content) in enumerate(files, start=1):
            row, _ = self.add_pdf_bytes(
                content=content,
                material_set_id=material["id"],
                file_name=file_name,
                order_index=index,
            )
            documents.append(row)
        return material, documents

    def _document_root(self, document: dict[str, Any]) -> Path:
        return self.paths.db_dir / "materials" / document["material_set_id"] / document["id"]

    def process_document(
        self,
        document_id: str,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        document = self.repo.get_document(document_id)
        if document is None:
            raise ValueError("找不到文档")
        material = self.repo.get_material_set(document["material_set_id"])
        if material is None:
            raise ValueError("找不到材料集")
        existing_pages = self.repo.list_pages(document_id)
        existing_sections = self.repo.list_sections(document_id)
        if existing_pages and existing_sections:
            return {"pages": len(existing_pages), "sections": len(existing_sections)}

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        self.repo.create_job(
            job_id=job_id,
            material_set_id=material["id"],
            document_id=document_id,
            stage="extract_split",
            total=max(document["page_count"], 1),
        )
        self.repo.update_document(document_id, extraction_status="running")
        try:
            pages = existing_pages
            if len(pages) < document["page_count"]:
                source = self.paths.db_dir / document["stored_path"]
                cached_pages = {
                    page["page_number"]: PageResult.model_validate(page)
                    for page in pages
                }

                def page_progress(current: int, total: int, message: str) -> None:
                    self.repo.update_job(
                        job_id,
                        completed=current,
                        message=message,
                    )
                    if progress:
                        progress(current, total, message)

                extract_pdf(
                    source,
                    page_progress,
                    existing_pages=cached_pages,
                    on_page=lambda page: self.repo.upsert_page(document_id, page),
                )
                pages = self.repo.list_pages(document_id)
                _write_jsonl(
                    self._document_root(document) / "extracted" / "pages.jsonl",
                    pages,
                )
                ocr_rows = [
                    page for page in pages if page["source_kind"] in {"ocr", "failed"}
                ]
                _write_jsonl(
                    self._document_root(document) / "ocr" / "pages.jsonl",
                    ocr_rows,
                )
            sections = existing_sections
            if not sections:
                split = split_pages(document_id=document_id, pages=pages)
                self.repo.replace_sections(document_id, split)
                sections = self.repo.list_sections(document_id)
                _write_json(
                    self._document_root(document) / "structure" / "sections.json",
                    sections,
                )

            docs = self.repo.list_documents(material["id"])
            inferred = _infer_metadata(
                material_type=material["material_type"],
                name=material["name"],
                file_names=[item["file_name"] for item in docs],
                page_texts=[page["final_text"] for page in pages[:8]],
            )
            updates = {
                key: value
                for key, value in inferred.items()
                if not material.get(key)
            }
            if updates:
                self.repo.update_material_metadata(
                    material["id"],
                    source="inferred:content",
                    **updates,
                )
            self.repo.update_document(
                document_id,
                extraction_status="succeeded",
                page_count=len(pages),
            )
            self.repo.update_job(
                job_id,
                status="succeeded",
                completed=len(pages),
                message=f"已生成 {len(sections)} 个章节块",
                finish=True,
            )
            self._write_manifest(material["id"])
            return {"pages": len(pages), "sections": len(sections)}
        except Exception as exc:
            self.repo.update_document(document_id, extraction_status="failed")
            self.repo.update_job(
                job_id,
                status="failed",
                message=str(exc),
                finish=True,
            )
            raise

    def process_material(
        self,
        material_set_id: str,
        progress: Callable[[str, int, int, str], None] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for document in self.repo.list_documents(material_set_id):
            def document_progress(current: int, total: int, message: str) -> None:
                if progress:
                    progress(document["file_name"], current, total, message)

            result = self.process_document(document["id"], document_progress)
            results.append({"document_id": document["id"], **result})
        return results

    def rebuild_sections(self, document_id: str) -> dict[str, Any]:
        document = self.repo.get_document(document_id)
        if document is None:
            raise ValueError("找不到文档")
        if self.repo.list_units(document_id=document_id):
            raise ValueError(
                "该文档已有结构化单元，不能直接重建章节；"
                "请先软删除相关单元或重新导入为新材料版本"
            )
        pages = self.repo.list_pages(document_id)
        if not pages:
            raise ValueError("该文档尚未完成文字提取")
        sections = split_pages(document_id=document_id, pages=pages)
        self.repo.replace_sections(document_id, sections)
        _write_json(
            self._document_root(document) / "structure" / "sections.json",
            self.repo.list_sections(document_id),
        )
        self._write_manifest(document["material_set_id"])
        return {"sections": len(sections)}

    def _ensure_sections(
        self,
        document: dict[str, Any],
        pages: list[dict[str, Any]],
        profile: ProviderProfile,
    ) -> list[dict[str, Any]]:
        sections = self.repo.list_sections(document["id"])
        if not needs_ai_heading_fallback(sections, len(pages)):
            return sections
        material = self.repo.get_material_set(document["material_set_id"])
        if material is None:
            raise ValueError("找不到材料集")
        client = OpenAICompatibleClient(profile)
        outline = client.infer_section_outline(
            material_type=material["material_type"],
            pages=pages,
        )
        fallback_sections: list[ExtractedSection] = []
        for index, item in enumerate(outline, start=1):
            page_group = [
                page
                for page in pages
                if item["page_start"] <= page["page_number"] <= item["page_end"]
            ]
            content = "\n\n".join(
                f"【第 {page['page_number']} 页】\n{page['final_text']}"
                for page in page_group
                if str(page.get("final_text") or "").strip()
            )
            fallback_sections.append(
                ExtractedSection(
                    id=(
                        f"{document['id']}_p{item['page_start']:03d}"
                        f"_s{index:03d}"
                    ),
                    document_id=document["id"],
                    path=item["title"],
                    title=item["title"],
                    level=1,
                    order_index=index,
                    page_start=item["page_start"],
                    page_end=item["page_end"],
                    content=content,
                    split_method="ai-fallback",
                )
            )
        self.repo.replace_sections(document["id"], fallback_sections)
        return self.repo.list_sections(document["id"])

    def structure_document(
        self,
        *,
        document_id: str,
        profile: ProviderProfile,
        force: bool = False,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        document = self.repo.get_document(document_id)
        if document is None:
            raise ValueError("找不到文档")
        material = self.repo.get_material_set(document["material_set_id"])
        if material is None:
            raise ValueError("找不到材料集")
        pages = self.repo.list_pages(document_id)
        if not pages:
            raise ValueError("请先完成文字提取")
        client = OpenAICompatibleClient(profile)
        sections = self._ensure_sections(document, pages, profile)
        page_texts = {page["page_number"]: page["final_text"] for page in pages}
        existing_by_section = {
            unit["section_id"]
            for unit in self.repo.list_units(document_id=document_id)
        }
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_root = self._document_root(document) / "runs" / run_id
        completed = 0
        failed = 0
        errors: list[str] = []
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        self.repo.create_job(
            job_id=job_id,
            material_set_id=material["id"],
            document_id=document_id,
            stage="ai_structure",
            total=len(sections),
        )
        for index, section in enumerate(sections, start=1):
            if section["id"] in existing_by_section and not force:
                completed += 1
                self.repo.update_job(job_id, completed=completed)
                continue
            try:
                result = client.structure_section(
                    material_type=material["material_type"],
                    section=section,
                    page_texts=page_texts,
                )
                raw_dir = run_root / "chunks"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / f"{section['id']}.raw.txt"
                raw_path.write_text(result.raw_response, encoding="utf-8")
                json_path = raw_dir / f"{section['id']}.json"
                _write_json(
                    json_path,
                    [unit.model_dump() for unit in result.units],
                )
                relative_raw = str(raw_path.relative_to(self.paths.db_dir))
                self.repo.save_units_for_section(
                    document_id=document_id,
                    section_id=section["id"],
                    units=result.units,
                    ai_profile=profile.vendor_name,
                    ai_base_url=profile.base_url,
                    ai_model=profile.model,
                    raw_response_path=relative_raw,
                )
                completed += 1
            except Exception as exc:
                failed += 1
                errors.append(f"{section['title']}: {exc}")
            if progress:
                progress(index, len(sections), section["title"])
            self.repo.update_job(
                job_id,
                completed=completed,
                failed=failed,
                message=errors[-1] if errors else None,
            )
        units = self.repo.list_units(document_id=document_id)
        _write_jsonl(
            self._document_root(document) / "structure" / "units.jsonl",
            units,
        )
        status = "succeeded" if failed == 0 else "partial"
        self.repo.update_job(
            job_id,
            status=status,
            completed=completed,
            failed=failed,
            message="；".join(errors[:5]) if errors else None,
            finish=True,
        )
        self._write_manifest(material["id"])
        return {
            "completed": completed,
            "failed": failed,
            "errors": errors,
            "run_id": run_id,
            "units": len(units),
        }

    def retry_units(
        self,
        *,
        unit_ids: Sequence[str],
        profile: ProviderProfile,
    ) -> dict[str, Any]:
        selected = []
        for unit_id in unit_ids:
            unit = self.repo.get_unit(unit_id)
            if unit and unit["deleted_at"] is None:
                selected.append(unit)
        section_ids = list(dict.fromkeys(unit["section_id"] for unit in selected))
        completed = 0
        errors: list[str] = []
        for section_id in section_ids:
            section = self.repo.get_section(section_id)
            if section is None:
                errors.append(f"{section_id}: 找不到章节")
                continue
            document = self.repo.get_document(section["document_id"])
            material = (
                self.repo.get_material_set(document["material_set_id"])
                if document
                else None
            )
            if document is None or material is None:
                errors.append(f"{section_id}: 找不到文档或材料集")
                continue
            pages = self.repo.list_pages(document["id"])
            page_texts = {page["page_number"]: page["final_text"] for page in pages}
            try:
                refreshed_section = dict(section)
                refreshed_section["content"] = "\n\n".join(
                    f"【第 {page['page_number']} 页】\n{page['final_text']}"
                    for page in pages
                    if section["page_start"]
                    <= page["page_number"]
                    <= section["page_end"]
                    and str(page["final_text"]).strip()
                )
                result = OpenAICompatibleClient(profile).structure_section(
                    material_type=material["material_type"],
                    section=refreshed_section,
                    page_texts=page_texts,
                )
                run_root = (
                    self._document_root(document)
                    / "runs"
                    / f"retry_{uuid.uuid4().hex[:12]}"
                )
                run_root.mkdir(parents=True, exist_ok=True)
                raw_path = run_root / "response.raw.txt"
                raw_path.write_text(result.raw_response, encoding="utf-8")
                self.repo.save_units_for_section(
                    document_id=document["id"],
                    section_id=section_id,
                    units=result.units,
                    ai_profile=profile.vendor_name,
                    ai_base_url=profile.base_url,
                    ai_model=profile.model,
                    raw_response_path=str(raw_path.relative_to(self.paths.db_dir)),
                )
                completed += 1
            except Exception as exc:
                errors.append(f"{section['title']}: {exc}")
        return {"completed": completed, "failed": len(errors), "errors": errors}

    def sync_material_exports(self, material_set_id: str) -> None:
        material = self.repo.get_material_set(material_set_id)
        if material is None:
            return
        for document in self.repo.list_documents(material_set_id):
            root = self._document_root(document)
            _write_jsonl(
                root / "structure" / "units.jsonl",
                self.repo.list_units(document_id=document["id"]),
            )
            _write_json(
                root / "structure" / "sections.json",
                self.repo.list_sections(document["id"]),
            )
        self._write_manifest(material_set_id)

    def _write_manifest(self, material_set_id: str) -> None:
        material = self.repo.get_material_set(material_set_id)
        if not material:
            return
        documents = self.repo.list_documents(material_set_id)
        manifest = {
            "material": material,
            "documents": documents,
        }
        _write_json(self.paths.materials_dir / material_set_id / "manifest.json", manifest)
