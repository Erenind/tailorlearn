from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf

from tailorlearn.models import PageResult, SourceKind

ProgressCallback = Callable[[int, int, str], None]

_OCR_ENGINE: Any | None = None


def _get_ocr_engine() -> Any:
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _meaningful_chars(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", text))


def text_is_usable(text: str) -> bool:
    meaningful = _meaningful_chars(text)
    if meaningful < 24:
        return False
    replacement_ratio = text.count("\ufffd") / max(len(text), 1)
    return replacement_ratio < 0.05


def _ocr_page(page: pymupdf.Page) -> tuple[str, float]:
    engine = _get_ocr_engine()
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
    image = cv2.imdecode(
        np.frombuffer(pixmap.tobytes("png"), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    result, _ = engine(image)
    if not result:
        return "", 0.0

    entries: list[tuple[float, float, str, float]] = []
    for item in result:
        box, text, score = item[0], str(item[1]), float(item[2])
        if not text.strip():
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        entries.append((min(ys), min(xs), text.strip(), score))
    entries.sort(key=lambda item: (round(item[0] / 8), item[1]))
    if not entries:
        return "", 0.0
    text = "\n".join(item[2] for item in entries)
    confidence = sum(item[3] for item in entries) / len(entries)
    return text, confidence


def page_as_png(pdf_path: Path, page_number: int, zoom: float = 1.8) -> bytes:
    with pymupdf.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError("页码超出文档范围")
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom),
            alpha=False,
        )
        return pixmap.tobytes("png")


def extract_pdf(
    pdf_path: Path,
    progress: ProgressCallback | None = None,
    existing_pages: dict[int, PageResult] | None = None,
    on_page: Callable[[PageResult], None] | None = None,
) -> list[PageResult]:
    results: list[PageResult] = []
    existing_pages = existing_pages or {}
    with pymupdf.open(pdf_path) as document:
        total = document.page_count
        for index, page in enumerate(document, start=1):
            cached = existing_pages.get(index)
            if cached is not None:
                results.append(cached)
                if progress:
                    progress(index, total, f"第 {index}/{total} 页（已缓存）")
                continue
            extracted = page.get_text("text").strip()
            if text_is_usable(extracted):
                text = extracted
                source_kind = SourceKind.TEXT
                confidence = None
                warning = None
                status = "ok"
            else:
                text, confidence = _ocr_page(page)
                source_kind = SourceKind.OCR
                warning = None
                status = "ok"
                if not text:
                    source_kind = SourceKind.FAILED
                    status = "failed"
                    warning = "OCR 未识别出文字"
                elif confidence < 0.65:
                    status = "warning"
                    warning = f"OCR 平均置信度较低：{confidence:.2f}"
                elif _meaningful_chars(text) < 12:
                    status = "warning"
                    warning = "识别出的有效文字过少"

            page_result = PageResult(
                page_number=index,
                raw_text=text,
                final_text=text,
                source_kind=source_kind,
                confidence=confidence,
                status=status,
                warning_reason=warning,
            )
            results.append(page_result)
            if on_page:
                on_page(page_result)
            if progress:
                progress(index, total, f"第 {index}/{total} 页")
    return results
