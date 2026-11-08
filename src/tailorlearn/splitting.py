from __future__ import annotations

import re
from typing import Any, Sequence

from tailorlearn.models import ExtractedSection

_FULLWIDTH_DIGITS = str.maketrans(
    "０１２３４５６７８９．\ue010",
    "0123456789..",
)
_CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十百零〇0-9]+章)\s*(.*)$")
_NUMERIC_HEADING_RE = re.compile(
    r"^([0-9]{1,2}(?:\.[0-9]{1,2}){1,2})\s*(.+)$"
)
_CN_MAJOR_RE = re.compile(r"^([一二三四五六七八九十]+、)\s*(.+)$")
_CN_SUB_RE = re.compile(r"^([（(][一二三四五六七八九十]+[）)])\s*(.+)$")
_APPENDIX_RE = re.compile(r"^(附录(?:[一二三四五六七八九十0-9]+)?)\s*(.*)$")
_FRONT_MATTER_RE = re.compile(r"^(前言|序言|目录|说明)\s*$")
_DOT_LEADER_RE = re.compile(r"(?:…{2,}|\.{4,}|·{4,})")


def _normalise_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.translate(_FULLWIDTH_DIGITS)).strip()


def _detect_heading(line: str) -> tuple[int, str] | None:
    if not line or len(line) > 90:
        return None
    if _DOT_LEADER_RE.search(line):
        return None
    if re.search(r"\s[0-9]{1,3}\s*$", line):
        # Running headers often repeat the chapter title followed by a page number.
        return None

    match = _CHAPTER_RE.match(line)
    if match:
        title = " ".join(part for part in match.groups() if part).strip()
        return 1, title

    match = _APPENDIX_RE.match(line)
    if match:
        title = " ".join(part for part in match.groups() if part).strip()
        return 1, title

    match = _FRONT_MATTER_RE.match(line)
    if match:
        return 1, match.group(1)

    match = _CN_MAJOR_RE.match(line)
    if match:
        return 1, f"{match.group(1)} {match.group(2)}".strip()

    match = _CN_SUB_RE.match(line)
    if match:
        return 2, f"{match.group(1)} {match.group(2)}".strip()

    match = _NUMERIC_HEADING_RE.match(line)
    if match:
        remainder = match.group(2).strip()
        if (
            not re.match(r"^[\u3400-\u9fff]", remainder)
            or len(remainder) > 40
            or re.search(r"[，。；：？！]", remainder)
        ):
            return None
        if re.match(r"^(?:dm|cm|mm|km|kg|m|h|min|s)\b", remainder):
            return None
        level = match.group(1).count(".") + 1
        return level, f"{match.group(1)} {remainder}".strip()
    return None


def split_pages(
    *,
    document_id: str,
    pages: Sequence[dict[str, Any]],
    method: str = "heading-rules",
) -> list[ExtractedSection]:
    sections: list[ExtractedSection] = []
    stack: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    used_paths: dict[str, int] = {}
    found_explicit_heading = False
    toc_pages = {
        int(page["page_number"])
        for page in pages
        if len(_DOT_LEADER_RE.findall(str(page.get("final_text") or ""))) >= 3
        or "目录" in str(page.get("final_text") or "").splitlines()[:3]
    }
    toc_candidates = {
        int(page["page_number"]): str(page.get("final_text") or "")
        for page in pages
    }
    changed = True
    while changed:
        changed = False
        for page_number in sorted(toc_candidates):
            if page_number in toc_pages or page_number - 1 not in toc_pages:
                continue
            lines = [
                line.strip()
                for line in toc_candidates[page_number].splitlines()
                if line.strip()
            ]
            numeric_only = sum(bool(re.fullmatch(r"[0-9]{1,3}", line)) for line in lines)
            if numeric_only >= 4 or _DOT_LEADER_RE.search(toc_candidates[page_number]):
                toc_pages.add(page_number)
                changed = True
    seen_generic_appendix = False

    def append_text(page_number: int, line: str) -> None:
        nonlocal current
        if current is None:
            current = {
                "title": f"第 {page_number} 页",
                "level": 1,
                "parent_id": None,
                "page_start": page_number,
                "page_end": page_number,
                "lines": [],
                "last_marker": None,
            }
        if current["last_marker"] != page_number:
            current["lines"].append(f"【第 {page_number} 页】")
            current["last_marker"] = page_number
        current["lines"].append(line)
        current["page_end"] = page_number

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        content = "\n".join(current["lines"]).strip()
        if content or current.get("explicit"):
            parent_titles = [item["title"] for item in stack[:-1]]
            raw_path = " / ".join([*parent_titles, current["title"]])
            occurrence = used_paths.get(raw_path, 0) + 1
            used_paths[raw_path] = occurrence
            path = raw_path if occurrence == 1 else f"{raw_path} [{occurrence}]"
            order_index = len(sections) + 1
            section_id = (
                f"{document_id}_p{current['page_start']:03d}_s{order_index:03d}"
            )
            sections.append(
                ExtractedSection(
                    id=section_id,
                    document_id=document_id,
                    parent_id=current["parent_id"],
                    path=path,
                    title=current["title"],
                    level=current["level"],
                    order_index=order_index,
                    page_start=current["page_start"],
                    page_end=current["page_end"],
                    content=content,
                    split_method=method,
                )
            )
        current = None

    for page in pages:
        page_number = int(page["page_number"])
        text = str(page.get("final_text") or "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if page_number in toc_pages:
            if current is None or current["title"] != "目录":
                flush()
                current = {
                    "title": "目录",
                    "level": 1,
                    "parent_id": None,
                    "page_start": page_number,
                    "page_end": page_number,
                    "lines": [],
                    "last_marker": None,
                    "explicit": True,
                }
            for line in lines:
                append_text(page_number, line)
            continue
        for line in lines:
            normalised = _normalise_line(line)
            heading = _detect_heading(normalised)
            if heading is None:
                append_text(page_number, line)
                continue

            level, title = heading
            if title == "附录":
                if seen_generic_appendix:
                    continue
                seen_generic_appendix = True
            if (
                level == 1
                and any(
                    item["level"] == 1 and item["title"] == title
                    for item in stack
                )
            ):
                # Ignore repeated running headers printed at the top of a page.
                continue
            found_explicit_heading = True
            flush()
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            parent_id = None
            if stack:
                parent_sections = [
                    section
                    for section in sections
                    if section.title == stack[-1]["title"]
                    and section.page_start == stack[-1]["page_start"]
                ]
                if parent_sections:
                    parent_id = parent_sections[-1].id
            current = {
                "title": title,
                "level": level,
                "parent_id": parent_id,
                "page_start": page_number,
                "page_end": page_number,
                "lines": [],
                "last_marker": None,
                "explicit": True,
            }
            stack.append(
                {
                    "title": title,
                    "level": level,
                    "page_start": page_number,
                }
            )
            append_text(page_number, line)
    flush()

    if not found_explicit_heading and pages:
        sections = []
        for page in pages:
            page_number = int(page["page_number"])
            text = str(page.get("final_text") or "").strip()
            if not text:
                continue
            order_index = len(sections) + 1
            sections.append(
                ExtractedSection(
                    id=f"{document_id}_p{page_number:03d}_s{order_index:03d}",
                    document_id=document_id,
                    path=f"第 {page_number} 页",
                    title=f"第 {page_number} 页",
                    level=1,
                    order_index=order_index,
                    page_start=page_number,
                    page_end=page_number,
                    content=f"【第 {page_number} 页】\n{text}",
                    split_method="page-fallback",
                )
            )
    return sections


def needs_ai_heading_fallback(
    sections: Sequence[ExtractedSection],
    page_count: int,
) -> bool:
    return page_count >= 12 and len(sections) <= 2
