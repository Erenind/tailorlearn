from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from tailorlearn.ai import OpenAICompatibleClient
from tailorlearn.db import Repository, utc_now
from tailorlearn.models import (
    OutlineDraftItem,
    OutlineDraftResponse,
    OutlineEvidence,
    OutlineItem,
    OutlineItemStatus,
    OutlineRelation,
    OutlineRelationType,
    OutlineRequirement,
    OutlineSourceKind,
    ProviderProfile,
)
from tailorlearn.paths import AppPaths

OUTLINE_SERVICE_VERSION = "2026-11-11-two-stage-outline-v3"


METHOD_LABELS = {
    "rule": "规则梳理",
    "ai": "AI 梳理",
    "ai_two_stage_textbook": "AI 梳理 · 课本模式",
    "ai_two_stage_theme": "AI 梳理 · 主题模式",
}
SCOPE_MODE_LABELS = {
    "textbook": "课本模式",
    "theme": "主题模式",
}
STATUS_LABELS = {
    OutlineItemStatus.DRAFT.value: "草稿",
    OutlineItemStatus.NEEDS_REVIEW.value: "待复核",
    OutlineItemStatus.VERIFIED.value: "已确认",
}
ITEM_TYPE_LABELS = {
    "knowledge_point": "知识要点",
    "skill": "技能方法",
    "application": "应用建模",
    "method": "思想方法",
}
RELATION_TYPE_LABELS = {
    OutlineRelationType.PREREQUISITE.value: "先备",
    OutlineRelationType.APPLICATION.value: "应用",
    OutlineRelationType.PARALLEL.value: "并列",
    OutlineRelationType.CONTRAST.value: "对比辨析",
    OutlineRelationType.OTHER.value: "其他",
}
SOURCE_KIND_LABELS = {
    OutlineSourceKind.STANDARD.value: "课程标准",
    OutlineSourceKind.TEXTBOOK.value: "教材",
    OutlineSourceKind.REQUIREMENT.value: "课程要求",
}
LEVEL_LABELS = {
    "了解": "了解",
    "理解": "理解",
    "掌握": "掌握",
    "运用": "运用",
    "经历": "经历",
}
_VERB_LEVELS = {
    "了解": ("了解", "了解"),
    "认识": ("认识", "了解"),
    "知道": ("知道", "了解"),
    "理解": ("理解", "理解"),
    "体会": ("体会", "理解"),
    "感悟": ("感悟", "理解"),
    "掌握": ("掌握", "掌握"),
    "能": ("能", "掌握"),
    "会": ("会", "掌握"),
    "运用": ("运用", "运用"),
    "应用": ("应用", "运用"),
    "探索": ("探索", "经历"),
    "经历": ("经历", "经历"),
    "形成": ("形成", "掌握"),
    "发展": ("发展", "掌握"),
}
_APPLICATION_MARKERS = ("解决", "实际", "应用", "解释", "列方程", "列不等式", "建模")
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_MARKER_RE = re.compile(
    rf"^\s*([{_CIRCLED}]|[0-9]{{1,2}}|[?？])\s*(\*?)(?=[\u4e00-\u9fff])"
)
_HEADING1_RE = re.compile(r"^\d{1,2}[.．]\s*([\u4e00-\u9fff][^，。；：\d]{1,16})$")
_HEADING2_RE = re.compile(
    r"^[（(]\d{1,2}[）)]\s*([\u4e00-\u9fff][^，。；：\d]{1,16})$"
)
_QUALIFIER_RE = re.compile(r"[一二三四五六七八九十两]元[一二三四五六七八九十]次")
_PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
_GENERIC_WORDS = (
    "进一步",
    "正确",
    "简单",
    "具体",
    "实际",
    "问题",
    "解释",
    "必要",
    "相应",
    "基本",
    "有关",
    "相关",
    "运用",
    "使用",
    "进行",
    "以及",
    "能够",
    "初步",
    "继续",
)
_LEADING_VERBS = (
    "能够",
    "进一步",
    "运用",
    "使用",
    "进行",
    "根据",
    "通过",
    "结合",
    "借助",
    "识别",
    "理解",
    "掌握",
    "了解",
    "认识",
    "探索",
    "经历",
    "体会",
    "感悟",
    "能",
    "会",
    "用",
    "把",
    "将",
    "对",
)


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)


def _bigrams(text: str) -> list[str]:
    normalised = _normalise(text)
    return [normalised[index : index + 2] for index in range(len(normalised) - 1)]


def _tokens(text: str) -> set[str]:
    return set(_bigrams(text))


def _lcs_ratio(needle: str, haystack: str) -> float:
    left = _normalise(needle)
    right = _normalise(haystack)
    if not left or not right:
        return 0.0
    if left in right:
        return 1.0
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1] / len(left)


def _lcs_length(needle: str, haystack: str) -> int:
    left = _normalise(needle)
    right = _normalise(haystack)
    if not left or not right:
        return 0
    if left in right:
        return len(left)
    previous = [0] * (len(right) + 1)
    best = 0
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
                best = max(best, previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return best


def _common_substring_ratio(needle: str, haystack: str) -> float:
    """Longest *contiguous* shared text, normalised by the needle length."""
    left = _normalise(needle)
    right = _normalise(haystack)
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    best = 0
    for left_char in left:
        current = [0] * (len(right) + 1)
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best / len(left)


def _split_phrases(text: str) -> list[str]:
    """Extract reusable content phrases from an indicator or a theme."""
    pieces: list[str] = []
    for raw_piece in re.split(r"[、，,；;：:（）()\s]+", text):
        piece = raw_piece.strip()
        if not piece:
            continue
        for part in re.split(r"(?:并且|以及|并|和|与|或)", piece):
            part = part.strip()
            for verb in _LEADING_VERBS:
                if part.startswith(verb) and len(part) > len(verb) + 1:
                    part = part[len(verb) :].strip()
                    break
            normalised = _normalise(part)
            for word in _GENERIC_WORDS:
                normalised = normalised.replace(word, "")
            if len(normalised) >= 2:
                pieces.append(normalised)
    if not pieces:
        normalised = _normalise(text)
        if normalised:
            pieces.append(normalised)
    return list(dict.fromkeys(pieces))


def _idf_table(documents: Sequence[str]) -> tuple[dict[str, float], float]:
    counts: Counter[str] = Counter()
    for document in documents:
        counts.update(_tokens(document))
    total = len(documents)
    table = {
        gram: math.log((total + 1) / (count + 0.5))
        for gram, count in counts.items()
    }
    return table, math.log((total + 1) / 0.5)


def _dice_idf(
    query: str,
    candidate: str,
    table: dict[str, float],
    default: float,
) -> float:
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0
    shared = query_tokens & candidate_tokens
    if not shared:
        return 0.0
    numerator = sum(table.get(gram, default) for gram in shared)
    denominator = min(
        sum(table.get(gram, default) for gram in query_tokens),
        sum(table.get(gram, default) for gram in candidate_tokens),
    )
    return numerator / denominator if denominator else 0.0


@dataclass
class Clause:
    page: int
    text: str
    printed_page: str | None = None
    marker: str = ""
    verb: str = ""
    level: str = ""
    optional: bool = False

    @property
    def body(self) -> str:
        stripped = self.text
        match = _MARKER_RE.match(stripped)
        if match:
            stripped = stripped[match.end() :]
        return stripped.strip()


@dataclass
class SectionRecord:
    id: str
    document_id: str
    material_set_id: str
    file_name: str
    order_index: int
    level: int
    title: str
    path: str
    page_start: int
    page_end: int
    chapter: str
    text: str
    score: float = 0.0
    theme_code: str = ""

    def as_scope_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.id,
            "document_id": self.document_id,
            "file_name": self.file_name,
            "level": self.level,
            "title": self.title,
            "path": self.path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chapter": self.chapter,
            "theme_code": self.theme_code,
            "match_score": round(self.score, 3),
        }


def _is_quality_section(section: dict[str, Any]) -> bool:
    title = str(section.get("title") or "").strip()
    if len(title) < 3:
        return False
    if title == "目录":
        return False
    if re.search(r"[＝=＋+\-－/\\]|犪|狓|狔", title):
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", title))
    return cjk >= 2


def _numeric_prefix(title: str) -> str:
    match = re.match(
        r"^(\d{1,2}(?:\.\d{1,2}){0,2})(?=\s|[\u4e00-\u9fff])",
        unicodedata.normalize("NFKC", title).strip(),
    )
    return match.group(1) if match else ""


def _printed_page(text: str) -> str | None:
    for line in reversed([item.strip() for item in text.splitlines() if item.strip()]):
        if _PAGE_NUMBER_RE.fullmatch(line):
            return line
        break
    return None


def _clause_from_text(page: int, text: str, printed_page: str | None) -> Clause:
    body = text.strip()
    marker = ""
    optional = False
    match = _MARKER_RE.match(body)
    if match:
        marker = match.group(1)
        optional = bool(match.group(2))
    elif re.match(r"^[（(]\d{1,2}[）)]", body):
        marker = body[:3]
    verb = ""
    level = ""
    head = body[match.end() :] if match else body
    for candidate in sorted(_VERB_LEVELS, key=len, reverse=True):
        position = head.find(candidate)
        if 0 <= position <= 12:
            verb, level = _VERB_LEVELS[candidate]
            break
    if "*" in body[:4]:
        optional = True
    return Clause(
        page=page,
        text=body,
        printed_page=printed_page,
        marker=marker,
        verb=verb,
        level=level,
        optional=optional,
    )


def _running_headers(pages: Sequence[dict[str, Any]]) -> set[str]:
    first_lines: Counter[str] = Counter()
    last_lines: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in str(page["final_text"]).splitlines() if line.strip()]
        if lines:
            first_lines[lines[0]] += 1
            last_lines[lines[-1]] += 1
    threshold = max(3, len(pages) // 4)
    headers = {
        line
        for line, count in first_lines.items()
        if count >= threshold and 0 < len(line) <= 40
    }
    headers |= {
        line
        for line, count in last_lines.items()
        if count >= threshold and 0 < len(line) <= 40 and not _PAGE_NUMBER_RE.fullmatch(line)
    }
    return headers


def _body_lines(page_text: str, headers: set[str]) -> list[str]:
    lines: list[str] = []
    for line in page_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line in headers or _PAGE_NUMBER_RE.fullmatch(line):
            continue
        lines.append(line)
    return lines


def _clauses_in_lines(page: int, lines: Sequence[str], printed: str | None) -> list[Clause]:
    body = "".join(lines)
    clauses: list[Clause] = []
    for part in re.split(r"(?<=[。；])", body):
        part = part.strip()
        if len(_normalise(part)) < 6:
            continue
        if part.startswith("[1]") or part.startswith("标有"):
            continue
        clause = _clause_from_text(page, part, printed)
        if (
            clauses
            and not clause.marker
            and clauses[-1].page == page
            and clauses[-1].text.endswith("；")
        ):
            previous = clauses[-1]
            clauses[-1] = Clause(
                page=page,
                text=previous.text + part,
                printed_page=printed,
                marker=previous.marker,
                verb=previous.verb,
                level=previous.level,
                optional=previous.optional,
            )
            continue
        clauses.append(clause)
    return clauses


def _section_scores(
    query: str,
    phrases: Sequence[str],
    candidate: str,
    table: dict[str, float],
    default: float,
) -> float:
    dice = _dice_idf(query, candidate, table, default)
    coverage = (
        sum(_lcs_ratio(phrase, candidate) for phrase in phrases) / len(phrases)
        if phrases
        else 0.0
    )
    return max(dice, 0.85 * coverage)


def _qualifier_conflict(query: str, candidate: str) -> bool:
    left = set(_QUALIFIER_RE.findall(query))
    right = set(_QUALIFIER_RE.findall(candidate))
    return bool(left and right and left.isdisjoint(right))


def _action_bigrams(phrases: Sequence[str]) -> set[str]:
    action_chars = set("列解求检代化去识移算证画判消除")
    grams: set[str] = set()
    for phrase in phrases:
        normalised = _normalise(phrase)
        for index in range(len(normalised) - 1):
            gram = normalised[index : index + 2]
            if any(character in action_chars for character in gram):
                grams.add(gram)
    return grams


def _first_sentence(text: str, haystack: str = "") -> str:
    body = re.sub(r"【第 \d+ 页】", "", text)
    for line in body.splitlines():
        line = line.strip()
        if len(_normalise(line)) < 8:
            continue
        for part in re.split(r"(?<=[。；])", line):
            part = part.strip()
            if len(_normalise(part)) >= 8:
                return part
    return body.strip()


def _quote_on_page(quote: str, page_text: str) -> bool:
    normalised = _normalise(quote)
    return bool(normalised) and normalised in _normalise(page_text)


class OutlineService:
    """Build, persist and export the curriculum outline for one dataset."""

    def __init__(self, paths: AppPaths, repository: Repository):
        self.paths = paths
        self.repo = repository

    @property
    def outlines_dir(self) -> Path:
        return self.paths.db_dir / "outlines"

    # ------------------------------------------------------------------
    # datasets, materials and scope
    # ------------------------------------------------------------------
    def list_datasets(self) -> list[dict[str, Any]]:
        with self.repo.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT * FROM learning_datasets
                    WHERE deleted_at IS NULL
                    ORDER BY created_at DESC, name
                    """
                ).fetchall()
            ]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self.repo.connect() as connection:
            row = connection.execute(
                "SELECT * FROM learning_datasets WHERE id=?",
                (dataset_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_requirements(
        self,
        dataset_id: str,
        theme_codes: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT * FROM learning_requirements
            WHERE dataset_id=? AND deleted_at IS NULL
        """
        params: list[Any] = [dataset_id]
        if theme_codes:
            placeholders = ",".join("?" for _ in theme_codes)
            query += f" AND theme_code IN ({placeholders})"
            params.extend(theme_codes)
        query += " ORDER BY theme_code, indicator_code"
        with self.repo.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(query, params).fetchall()
            ]

    def list_themes(
        self,
        dataset_id: str,
    ) -> list[dict[str, Any]]:
        themes: dict[str, dict[str, Any]] = {}
        for row in self.list_requirements(dataset_id):
            code = row.get("theme_code") or "未编号"
            theme = themes.setdefault(
                code,
                {
                    "theme_code": code,
                    "theme_name": row.get("theme_name") or "",
                    "theme_goal": row.get("theme_goal") or "",
                    "indicators": [],
                },
            )
            theme["indicators"].append(
                {
                    "indicator_code": row["indicator_code"],
                    "indicator_name": row.get("indicator_name") or "",
                    "standard_ref": row.get("standard_ref") or "",
                    "standard_page": row.get("standard_page") or "",
                    "textbook_page": row.get("textbook_page") or "",
                }
            )
        return list(themes.values())

    def materials(self, material_type: str) -> list[dict[str, Any]]:
        return self.repo.list_material_sets(material_type=material_type)

    def documents(self, material_set_id: str) -> list[dict[str, Any]]:
        return self.repo.list_documents(material_set_id)

    def _standard_page_window(
        self,
        requirements: Sequence[dict[str, Any]],
    ) -> tuple[int | None, int | None]:
        pages: list[int] = []
        for row in requirements:
            for part in re.split(r"[／/、,，\s]+", str(row.get("standard_page") or "")):
                if part.isdigit():
                    pages.append(int(part))
        if not pages:
            return None, None
        return min(pages), max(pages)

    def _headings(
        self,
        pages: Sequence[dict[str, Any]],
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> list[tuple[int, int, int, str]]:
        headers = _running_headers(pages)
        headings: list[tuple[int, int, int, str]] = []
        for page in pages:
            number = int(page["page_number"])
            if page_start is not None and number < page_start:
                continue
            if page_end is not None and number > page_end:
                continue
            for index, line in enumerate(_body_lines(page["final_text"], headers)):
                for level, pattern in ((1, _HEADING1_RE), (2, _HEADING2_RE)):
                    match = pattern.match(line)
                    if match:
                        headings.append((number, index, level, match.group(1).strip()))
                        break
        return headings

    def _standard_blocks(
        self,
        pages: Sequence[dict[str, Any]],
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> list[dict[str, Any]]:
        headers = _running_headers(pages)
        page_lines = {
            int(page["page_number"]): _body_lines(page["final_text"], headers)
            for page in pages
        }
        headings = self._headings(pages, page_start, page_end)
        if not headings:
            first = page_start or min(page_lines)
            last = page_end or max(page_lines)
            clauses: list[Clause] = []
            for number in sorted(page_lines):
                if first <= number <= last:
                    printed = _printed_page(
                        next(
                            page["final_text"]
                            for page in pages
                            if int(page["page_number"]) == number
                        )
                    )
                    clauses.extend(
                        _clauses_in_lines(number, page_lines[number], printed)
                    )
            return [
                {
                    "title": "未识别章节",
                    "level": 1,
                    "page_start": first,
                    "page_end": last,
                    "clauses": clauses,
                }
            ]
        blocks: list[dict[str, Any]] = []
        for index, (page, line_index, level, title) in enumerate(headings):
            end_page = max(page_lines)
            end_index = len(page_lines[end_page])
            for page2, line2, level2, _ in headings[index + 1 :]:
                if level2 <= level:
                    end_page, end_index = page2, line2
                    break
            clauses: list[Clause] = []
            for number in sorted(page_lines):
                if number < page or number > end_page:
                    continue
                lines = page_lines[number]
                start = line_index + 1 if number == page else 0
                stop = end_index if number == end_page else len(lines)
                if number == page and start >= len(lines):
                    continue
                printed = _printed_page(
                    next(
                        page_text["final_text"]
                        for page_text in pages
                        if int(page_text["page_number"]) == number
                    )
                )
                clauses.extend(
                    _clauses_in_lines(number, lines[start:stop], printed)
                )
            if not clauses:
                continue
            blocks.append(
                {
                    "title": title,
                    "level": level,
                    "page_start": min(clause.page for clause in clauses),
                    "page_end": max(clause.page for clause in clauses),
                    "clauses": clauses,
                }
            )
        return blocks

    def _block_for_theme(
        self,
        blocks: Sequence[dict[str, Any]],
        theme: dict[str, Any],
    ) -> dict[str, Any] | None:
        theme_name = _normalise(theme["theme_name"])
        contained = [
            block
            for block in blocks
            if block["level"] == 2
            and _normalise(block["title"])
            and _normalise(block["title"]) in theme_name
        ]
        if contained:
            return min(contained, key=lambda block: len(block["title"]))
        contained = [
            block
            for block in blocks
            if _normalise(block["title"])
            and _normalise(block["title"]) in theme_name
        ]
        if contained:
            return min(contained, key=lambda block: (block["level"], len(block["title"])))
        scored = [
            (
                _lcs_ratio(block["title"], theme["theme_name"]),
                block,
            )
            for block in blocks
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored and scored[0][0] >= 0.5:
            return scored[0][1]
        return None

    def _textbook_records(
        self,
        material_set_id: str,
        document_ids: Sequence[str],
    ) -> list[SectionRecord]:
        records: list[SectionRecord] = []
        material = self.repo.get_material_set(material_set_id)
        material_name = material["name"] if material else material_set_id
        for document in self.repo.list_documents(material_set_id):
            if document_ids and document["id"] not in document_ids:
                continue
            pages = self.repo.list_pages(document["id"])
            page_texts = {
                int(page["page_number"]): page["final_text"] for page in pages
            }
            sections = self.repo.list_sections(document["id"])
            levels = {int(section["level"]) for section in sections}
            if 2 in levels:
                wanted_level = 2
            elif 1 in levels:
                wanted_level = 1
            else:
                wanted_level = min(levels) if levels else 1
            chapter = ""
            chapters: dict[str, str] = {}
            for section in sections:
                if section["level"] == 1:
                    chapter = section["title"]
                chapters[section["id"]] = chapter
            for index, section in enumerate(sections):
                if int(section["level"]) != wanted_level:
                    continue
                if not _is_quality_section(section):
                    continue
                end = int(section["page_end"])
                prefix = _numeric_prefix(section["title"])
                if prefix:
                    for later in sections:
                        later_prefix = _numeric_prefix(later["title"])
                        if later_prefix.startswith(f"{prefix}."):
                            end = max(end, int(later["page_end"]))
                else:
                    for later in sections[index + 1 :]:
                        if later["level"] <= section["level"]:
                            break
                        end = max(end, int(later["page_end"]))
                text = "\n".join(
                    page_texts[number]
                    for number in range(int(section["page_start"]), end + 1)
                    if number in page_texts
                )
                records.append(
                    SectionRecord(
                        id=section["id"],
                        document_id=document["id"],
                        material_set_id=material_set_id,
                        file_name=document["file_name"],
                        order_index=document["order_index"],
                        level=int(section["level"]),
                        title=section["title"],
                        path=section["path"],
                        page_start=int(section["page_start"]),
                        page_end=end,
                        chapter=chapters.get(section["id"], section["title"]),
                        text=text,
                    )
                )
        return records

    def _theme_anchors(self, theme: dict[str, Any]) -> list[str]:
        anchors: list[str] = []
        suffixes = (
            "运算",
            "加减",
            "应用",
            "表示",
            "解法",
            "性质",
            "概念",
            "方法",
            "意义",
        )
        for part in re.split(r"(?:与|和|、|及)", theme["theme_name"]):
            normalised = _normalise(part)
            if len(normalised) >= 2:
                anchors.append(normalised)
                for suffix in suffixes:
                    if (
                        normalised.endswith(suffix)
                        and len(normalised) > len(suffix) + 1
                    ):
                        anchors.append(normalised[: -len(suffix)])
        if not anchors:
            anchors.append(_normalise(theme["theme_name"]))
        return list(dict.fromkeys(anchors))

    def _match_sections(
        self,
        themes: Sequence[dict[str, Any]],
        records: Sequence[SectionRecord],
    ) -> tuple[dict[str, list[SectionRecord]], list[SectionRecord]]:
        selected: dict[str, list[SectionRecord]] = defaultdict(list)
        used: set[str] = set()
        for theme in themes:
            anchors = self._theme_anchors(theme)
            query_parts = [
                theme["theme_name"],
                theme["theme_goal"],
                *[item["indicator_name"] for item in theme["indicators"]],
            ]
            query = " ".join(query_parts)
            phrases = _split_phrases(query)
            documents = [record.text[:3000] for record in records] + [query]
            table, default = _idf_table(documents)
            candidates: list[SectionRecord] = []
            for record in records:
                if record.id in used:
                    continue
                blob = f"{record.title} {record.chapter} {record.text[:3000]}"
                if not any(anchor in record.title or anchor in record.chapter for anchor in anchors):
                    continue
                score = _section_scores(query, phrases, blob, table, default)
                if score < 0.18:
                    continue
                candidates.append(
                    SectionRecord(**{**record.__dict__, "score": round(score, 3)})
                )
            candidates.sort(key=lambda record: record.score, reverse=True)
            if candidates:
                best = candidates[0].score
                for record in candidates:
                    if record.score >= max(0.18, best * 0.55):
                        record.theme_code = theme["theme_code"]
                        selected[theme["theme_code"]].append(record)
                        used.add(record.id)
        return selected, [record for record in records if record.id in used]

    def suggest_sections(
        self,
        *,
        dataset_id: str,
        theme_codes: Sequence[str],
        textbook_material_id: str,
        textbook_document_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        themes = [
            theme
            for theme in self.list_themes(dataset_id)
            if not theme_codes or theme["theme_code"] in theme_codes
        ]
        records = self._textbook_records(
            textbook_material_id,
            list(textbook_document_ids),
        )
        selected, _ = self._match_sections(themes, records)
        suggestions: list[dict[str, Any]] = []
        for theme in themes:
            for record in selected.get(theme["theme_code"], []):
                entry = record.as_scope_dict()
                entry["theme_name"] = theme["theme_name"]
                suggestions.append(entry)
        suggestions.sort(key=lambda item: (item["theme_code"], item["page_start"]))
        return suggestions

    def section_options(
        self,
        textbook_material_id: str,
        textbook_document_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        records = self._textbook_records(
            textbook_material_id,
            list(textbook_document_ids),
        )
        options = [record.as_scope_dict() for record in records]
        options.sort(key=lambda item: (item["file_name"], item["page_start"]))
        return options

    def build_scope(
        self,
        *,
        dataset_id: str,
        theme_codes: Sequence[str],
        standard_material_id: str,
        standard_document_id: str,
        textbook_material_id: str,
        textbook_document_ids: Sequence[str] = (),
        section_ids: Sequence[str] | None = None,
        standard_page_start: int | None = None,
        standard_page_end: int | None = None,
    ) -> dict[str, Any]:
        dataset = self.get_dataset(dataset_id)
        if dataset is None:
            raise ValueError("找不到学情数据集")
        themes = [
            theme
            for theme in self.list_themes(dataset_id)
            if not theme_codes or theme["theme_code"] in theme_codes
        ]
        if not themes:
            raise ValueError("所选主题在课程要求中不存在")
        standard_material = self.repo.get_material_set(standard_material_id)
        if standard_material is None:
            raise ValueError("找不到课程标准材料")
        standard_document = self.repo.get_document(standard_document_id)
        if standard_document is None or standard_document["material_set_id"] != standard_material_id:
            raise ValueError("课程标准文档不属于所选材料")
        textbook_material = self.repo.get_material_set(textbook_material_id)
        if textbook_material is None:
            raise ValueError("找不到教材材料")
        requirements = self.list_requirements(
            dataset_id,
            [theme["theme_code"] for theme in themes],
        )
        window_start, window_end = self._standard_page_window(requirements)
        if standard_page_start:
            window_start = standard_page_start
        if standard_page_end:
            window_end = standard_page_end
        standard_pages = self.repo.list_pages(standard_document_id)
        window_start = window_start or min(
            int(page["page_number"]) for page in standard_pages
        )
        window_end = window_end or max(
            int(page["page_number"]) for page in standard_pages
        )
        blocks = self._standard_blocks(
            standard_pages,
            page_start=max(1, window_start - 5),
            page_end=min(len(standard_pages), window_end + 5),
        )
        standard_themes: list[dict[str, Any]] = []
        warnings: list[str] = []
        for theme in themes:
            block = self._block_for_theme(blocks, theme)
            if block is None:
                warnings.append(
                    f"未能在课程标准第 {window_start}–{window_end} 页附近"
                    f"识别与“{theme['theme_name']}”对应的内容要求标题，"
                    "请人工指定页码范围。"
                )
                standard_themes.append(
                    {
                        "theme_code": theme["theme_code"],
                        "theme_name": theme["theme_name"],
                        "block_title": "",
                        "level": 0,
                        "page_start": window_start,
                        "page_end": window_end,
                        "clause_count": 0,
                        "matched": False,
                    }
                )
                continue
            standard_themes.append(
                {
                    "theme_code": theme["theme_code"],
                    "theme_name": theme["theme_name"],
                    "block_title": block["title"],
                    "level": block["level"],
                    "page_start": block["page_start"],
                    "page_end": block["page_end"],
                    "clause_count": len(block["clauses"]),
                    "matched": True,
                }
            )
        records = self._textbook_records(
            textbook_material_id,
            list(textbook_document_ids),
        )
        if section_ids:
            allowed = {record.id for record in records}
            missing = [section_id for section_id in section_ids if section_id not in allowed]
            if missing:
                raise ValueError("所选章节不属于当前教材材料或不是可梳理小节")
            chosen: dict[str, list[SectionRecord]] = defaultdict(list)
            theme_by_code = {theme["theme_code"]: theme for theme in themes}
            for record in records:
                if record.id not in section_ids:
                    continue
                best_theme = ""
                best_score = -1.0
                for theme in themes:
                    query = " ".join(
                        [
                            theme["theme_name"],
                            theme["theme_goal"],
                            *[item["indicator_name"] for item in theme["indicators"]],
                        ]
                    )
                    phrases = _split_phrases(query)
                    blob = f"{record.title} {record.chapter} {record.text[:2000]}"
                    table, default = _idf_table([record.text[:2000], query])
                    score = _section_scores(query, phrases, blob, table, default)
                    if score > best_score:
                        best_score = score
                        best_theme = theme["theme_code"]
                record.theme_code = best_theme
                record.score = round(max(best_score, 0.0), 3)
                if best_theme in theme_by_code:
                    chosen[best_theme].append(record)
            selected = chosen
            selected_records = [
                record for record in records if record.id in set(section_ids)
            ]
        else:
            selected, selected_records = self._match_sections(themes, records)
        selected_sections: list[dict[str, Any]] = []
        for theme in themes:
            for record in selected.get(theme["theme_code"], []):
                entry = record.as_scope_dict()
                entry["theme_name"] = theme["theme_name"]
                selected_sections.append(entry)
        selected_sections.sort(key=lambda item: (item["theme_code"], item["page_start"]))
        if not selected_sections:
            warnings.append(
                "未能在所选教材切片中匹配到与主题相关的章节，"
                "请手动选择章节或调整主题范围。"
            )
        excluded = self._excluded_sections(records, selected_records)
        standard_pages_all = sorted(
            {
                page
                for entry in standard_themes
                if entry["matched"]
                for page in range(entry["page_start"], entry["page_end"] + 1)
            }
        )
        printed_pages = sorted(
            {
                clause.printed_page
                for entry in standard_themes
                if entry["matched"]
                for clause in next(
                    (
                        block["clauses"]
                        for block in blocks
                        if block["title"] == entry["block_title"]
                        and block["page_start"] == entry["page_start"]
                    ),
                    [],
                )
                if clause.printed_page
            },
            key=lambda value: int(value),
        )
        book_names = sorted(
            {record.file_name for record in selected_records}
        )
        statement = (
            f"本次梳理范围：学情数据集“{dataset['name']}”；"
            f"主题 {len(themes)} 个（"
            + "、".join(
                f"{theme['theme_code']} {theme['theme_name']}" for theme in themes
            )
            + f"）；观察指标 {sum(len(theme['indicators']) for theme in themes)} 项；"
            f"课程标准“{standard_material['name']}”"
            + (
                "内容要求"
                + "、".join(
                    f"“{entry['block_title']}”" for entry in standard_themes if entry["matched"]
                )
                if any(entry["matched"] for entry in standard_themes)
                else "内容要求（按页码范围）"
            )
            + (
                f"（文件第 {standard_pages_all[0]}–{standard_pages_all[-1]} 页"
                if standard_pages_all
                else "（页码待确认"
            )
            + (
                f"，正文第 {printed_pages[0]}–{printed_pages[-1]} 页）"
                if printed_pages
                else "）"
            )
            + f"；教材“{textbook_material['name']}”"
            f"{len(book_names)} 份切片、{len({item['chapter'] for item in selected_sections})} 个章节、"
            f"{len(selected_sections)} 个小节"
            + (f"（{'、'.join(book_names)}）" if book_names else "")
            + "。"
        )
        excluded_note = ""
        if excluded:
            excluded_note = (
                "已确定主题范围，其余内容未纳入本次梳理："
                + "；".join(
                    f"{entry['chapter']}（{entry['file_name']}，"
                    f"第 {entry['page_start']}–{entry['page_end']} 页，"
                    f"{entry['section_count']} 个小节）"
                    for entry in excluded[:6]
                )
                + ("等。" if len(excluded) > 6 else "。")
            )
        scope = {
            "dataset": {
                "id": dataset["id"],
                "name": dataset["name"],
                "description": dataset.get("description") or "",
            },
            "themes": [
                {
                    "theme_code": theme["theme_code"],
                    "theme_name": theme["theme_name"],
                    "theme_goal": theme["theme_goal"],
                    "indicators": theme["indicators"],
                }
                for theme in themes
            ],
            "standard": {
                "material_set_id": standard_material["id"],
                "material_name": standard_material["name"],
                "document_id": standard_document["id"],
                "file_name": standard_document["file_name"],
                "page_window": [window_start, window_end],
                "themes": standard_themes,
            },
            "textbook": {
                "material_set_id": textbook_material["id"],
                "material_name": textbook_material["name"],
                "file_names": book_names,
                "sections": selected_sections,
                "excluded": excluded,
            },
            "boundary": {
                "statement": statement,
                "excluded_note": excluded_note,
                "warnings": warnings,
            },
            "created_at": utc_now(),
        }
        return scope

    def _excluded_sections(
        self,
        records: Sequence[SectionRecord],
        selected: Sequence[SectionRecord],
    ) -> list[dict[str, Any]]:
        selected_ids = {record.id for record in selected}
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            if record.id in selected_ids:
                continue
            key = (record.file_name, record.chapter)
            entry = grouped.setdefault(
                key,
                {
                    "file_name": record.file_name,
                    "chapter": record.chapter,
                    "page_start": record.page_start,
                    "page_end": record.page_end,
                    "section_count": 0,
                    "titles": [],
                },
            )
            entry["page_start"] = min(entry["page_start"], record.page_start)
            entry["page_end"] = max(entry["page_end"], record.page_end)
            entry["section_count"] += 1
            if len(entry["titles"]) < 4:
                entry["titles"].append(record.title)
        return sorted(
            grouped.values(),
            key=lambda item: (item["file_name"], item["page_start"]),
        )

    def _scope_standard_clauses(
        self,
        scope: dict[str, Any],
    ) -> dict[str, list[Clause]]:
        """Re-derive the clause list of each theme block from stored pages."""
        document_id = scope["standard"]["document_id"]
        pages = self.repo.list_pages(document_id)
        headers = _running_headers(pages)
        page_lines = {
            int(page["page_number"]): _body_lines(page["final_text"], headers)
            for page in pages
        }
        blocks = self._standard_blocks(pages)
        result: dict[str, list[Clause]] = {}
        for entry in scope["standard"]["themes"]:
            block = next(
                (
                    candidate
                    for candidate in blocks
                    if candidate["title"] == entry["block_title"]
                    and candidate["page_start"] == entry["page_start"]
                ),
                None,
            )
            if block is None:
                clauses: list[Clause] = []
                for number in range(entry["page_start"], entry["page_end"] + 1):
                    if number in page_lines:
                        clauses.extend(
                            _clauses_in_lines(
                                number,
                                page_lines[number],
                                _printed_page(
                                    next(
                                        page["final_text"]
                                        for page in pages
                                        if int(page["page_number"]) == number
                                    )
                                ),
                            )
                        )
                result[entry["theme_code"]] = clauses
            else:
                result[entry["theme_code"]] = list(block["clauses"])
        return result

    def _scope_section_records(
        self,
        scope: dict[str, Any],
    ) -> dict[str, SectionRecord]:
        records: dict[str, SectionRecord] = {}
        material_id = scope["textbook"]["material_set_id"]
        for entry in scope["textbook"]["sections"]:
            document = self.repo.get_document(entry["document_id"])
            if document is None:
                continue
            pages = self.repo.list_pages(entry["document_id"])
            text = "\n".join(
                page["final_text"]
                for page in pages
                if entry["page_start"] <= int(page["page_number"]) <= entry["page_end"]
            )
            records[entry["section_id"]] = SectionRecord(
                id=entry["section_id"],
                document_id=entry["document_id"],
                material_set_id=material_id,
                file_name=entry["file_name"],
                order_index=0,
                level=entry["level"],
                title=entry["title"],
                path=entry["path"],
                page_start=entry["page_start"],
                page_end=entry["page_end"],
                chapter=entry["chapter"],
                text=text,
                score=entry.get("match_score", 0.0),
                theme_code=entry.get("theme_code", ""),
            )
        return records

    # ------------------------------------------------------------------
    # item generation
    # ------------------------------------------------------------------
    def _item_type(self, content: str) -> str:
        if any(marker in content for marker in ("解决", "实际", "应用", "解释", "建模")):
            return "application"
        if any(
            marker in content
            for marker in (
                "性质",
                "法则",
                "运算",
                "化简",
                "求解",
                "计算",
                "解集",
                "解方程",
                "解一元",
                "解二元",
                "解法",
            )
        ):
            return "skill"
        if any(marker in content for marker in ("思想", "方法", "策略")):
            return "method"
        return "knowledge_point"

    def _evidence_from_clause(
        self,
        clause: Clause,
        scope: dict[str, Any],
    ) -> OutlineEvidence:
        standard = scope["standard"]
        return OutlineEvidence(
            kind=OutlineSourceKind.STANDARD.value,
            material_set_id=standard["material_set_id"],
            material_name=standard["material_name"],
            document_id=standard["document_id"],
            file_name=standard["file_name"],
            section_path="",
            page=clause.page,
            printed_page=clause.printed_page,
            quote=clause.text,
            verified=True,
            verification_note="引文来自课程标准逐页提取文本",
        )

    def _pick_quote(
        self,
        page_texts: dict[int, str],
        page_start: int,
        page_end: int,
        phrase: str,
    ) -> tuple[int, str]:
        normalised_phrase = _normalise(phrase)
        fallback: tuple[int, str] | None = None
        for number in range(page_start, page_end + 1):
            text = page_texts.get(number, "")
            for line in text.splitlines():
                line = line.strip()
                if not line or _PAGE_NUMBER_RE.fullmatch(line):
                    continue
                normalised_line = unicodedata.normalize("NFKC", line)
                if re.match(
                    r"^第[一二三四五六七八九十百零〇\d]+章",
                    normalised_line,
                ):
                    continue
                if re.match(r"^\d{1,2}(?:\.\d{1,2}){1,2}\s", normalised_line):
                    continue
                if re.match(
                    r"^\d{1,2}(?:\.\d{1,2}){1,2}\s*[\u4e00-\u9fff]",
                    normalised_line,
                ):
                    continue
                for part in re.split(r"(?<=[。；])", line):
                    part = part.strip()
                    length = len(_normalise(part))
                    if length < 10:
                        continue
                    if "。" not in part and length < 22:
                        continue
                    if normalised_phrase and normalised_phrase in _normalise(part):
                        return number, part[:160]
                    if fallback is None and length >= 24:
                        fallback = (number, part[:160])
        if fallback is not None:
            return fallback
        for number in range(page_start, page_end + 1):
            text = _first_sentence(page_texts.get(number, ""))
            if len(_normalise(text)) >= 10:
                return number, text[:160]
        return page_start, ""

    def _unit_evidence(
        self,
        record: SectionRecord,
        scope: dict[str, Any],
        phrases: Sequence[str],
    ) -> list[tuple[float, OutlineEvidence]]:
        units = self.repo.list_units(document_id=record.document_id)
        pages = {
            int(page["page_number"]): page["final_text"]
            for page in self.repo.list_pages(record.document_id)
        }
        results: list[tuple[float, OutlineEvidence]] = []
        for unit in units:
            if unit["status"] == "draft" and unit["unit_type"] in {"exercise", "other"}:
                continue
            if unit["unit_type"] not in {
                "concept",
                "property",
                "theorem",
                "summary",
                "activity",
            }:
                continue
            if not (record.page_start <= int(unit.get("page_start", 0) or 0) <= record.page_end):
                refs = []
                try:
                    refs = json.loads(unit.get("source_refs_json") or "[]")
                except json.JSONDecodeError:
                    refs = []
                if not any(
                    record.page_start <= int(ref.get("page") or 0) <= record.page_end
                    for ref in refs
                ):
                    continue
            title = str(unit["title"]).strip()
            if title.startswith("例") or len(_normalise(title)) < 2:
                continue
            score = max(
                (
                    _lcs_ratio(phrase, title)
                    for phrase in phrases
                ),
                default=0.0,
            )
            if score < 0.45:
                continue
            refs = []
            try:
                refs = json.loads(unit.get("source_refs_json") or "[]")
            except json.JSONDecodeError:
                refs = []
            page = None
            quote = ""
            for ref in refs:
                candidate_page = int(ref.get("page") or 0)
                candidate_quote = str(ref.get("quote") or "").strip()
                if (
                    candidate_page
                    and record.page_start <= candidate_page <= record.page_end
                    and candidate_quote
                ):
                    page = candidate_page
                    quote = candidate_quote[:160]
                    break
            if page is None:
                page = record.page_start
            if not quote:
                quote = title
            verified = _quote_on_page(quote, pages.get(page, ""))
            evidence = OutlineEvidence(
                kind=OutlineSourceKind.TEXTBOOK.value,
                material_set_id=scope["textbook"]["material_set_id"],
                material_name=scope["textbook"]["material_name"],
                document_id=record.document_id,
                file_name=record.file_name,
                section_id=record.id,
                section_path=f"{record.chapter} / {record.title}",
                page=page,
                page_end=record.page_end,
                printed_page=None,
                quote=quote,
                unit_id=unit["id"],
                verified=verified,
                verification_note=(
                    "引文来自已结构化单元并回查该页文本"
                    if verified
                    else "单元引文未能在该页提取文本中逐字匹配"
                ),
            )
            results.append((score, evidence))
        results.sort(key=lambda item: item[0], reverse=True)
        return results

    def _section_evidence(
        self,
        record: SectionRecord,
        scope: dict[str, Any],
        phrase: str,
    ) -> OutlineEvidence:
        pages = self.repo.list_pages(record.document_id)
        page_texts = {
            int(page["page_number"]): page["final_text"] for page in pages
        }
        page, quote = self._pick_quote(
            page_texts,
            record.page_start,
            record.page_end,
            phrase,
        )
        verified = bool(quote) and _quote_on_page(quote, page_texts.get(page, ""))
        return OutlineEvidence(
            kind=OutlineSourceKind.TEXTBOOK.value,
            material_set_id=scope["textbook"]["material_set_id"],
            material_name=scope["textbook"]["material_name"],
            document_id=record.document_id,
            file_name=record.file_name,
            section_id=record.id,
            section_path=f"{record.chapter} / {record.title}",
            page=page,
            page_end=record.page_end,
            quote=quote,
            verified=verified,
            verification_note=(
                "引文回查教材该页提取文本成功"
                if verified
                else "未能在该小节页范围内取得可核对引文"
            ),
        )

    def _key_points(
        self,
        record: SectionRecord,
        scope: dict[str, Any],
    ) -> list[str]:
        points: list[str] = []
        sections = self.repo.list_sections(record.document_id)
        inside = [
            section
            for section in sections
            if section["level"] > record.level
            and record.page_start <= int(section["page_start"]) <= record.page_end
        ]
        for section in inside:
            if _is_quality_section(section):
                points.append(section["title"])
        units = self.repo.list_units(document_id=record.document_id)
        for unit in units:
            if unit["unit_type"] not in {"concept", "property", "theorem", "summary"}:
                continue
            refs = []
            try:
                refs = json.loads(unit.get("source_refs_json") or "[]")
            except json.JSONDecodeError:
                refs = []
            pages = [int(ref.get("page") or 0) for ref in refs]
            if not any(record.page_start <= page <= record.page_end for page in pages):
                continue
            title = str(unit["title"]).strip()
            if title.startswith("例") or len(_normalise(title)) < 2:
                continue
            points.append(title)
        return list(dict.fromkeys(points))

    def rule_items(self, scope: dict[str, Any]) -> list[OutlineItem]:
        clauses_by_theme = self._scope_standard_clauses(scope)
        records = self._scope_section_records(scope)
        theme_sections: dict[str, list[SectionRecord]] = defaultdict(list)
        for entry in scope["textbook"]["sections"]:
            record = records.get(entry["section_id"])
            if record is not None:
                theme_sections[entry["theme_code"]].append(record)
        section_documents = [
            f"{record.title} {record.text[:2500]}"
            for record in records.values()
        ]
        items: list[OutlineItem] = []
        for theme in scope["themes"]:
            theme_clauses = clauses_by_theme.get(theme["theme_code"], [])
            indicator_names = [
                item["indicator_name"] for item in theme["indicators"]
            ]
            clause_table, clause_default = _idf_table(
                [clause.text for clause in theme_clauses] + indicator_names
            )
            section_table, section_default = _idf_table(
                section_documents + indicator_names + [theme["theme_name"]]
            )
            for indicator in theme["indicators"]:
                content = indicator["indicator_name"].strip()
                phrases = _split_phrases(content)
                query = " ".join([theme["theme_name"], content])
                scored_clauses: list[tuple[float, float, Clause]] = []
                for clause in theme_clauses:
                    if clause.optional:
                        continue
                    score = _section_scores(
                        query,
                        phrases,
                        clause.body,
                        clause_table,
                        clause_default,
                    )
                    conflict = _qualifier_conflict(content, clause.body)
                    if conflict:
                        score *= 0.15
                    best_lcs = max(
                        (_lcs_ratio(phrase, clause.body) for phrase in phrases),
                        default=0.0,
                    )
                    matched = max(
                        (_lcs_length(phrase, clause.body) for phrase in phrases),
                        default=0,
                    )
                    if conflict or matched < 2:
                        best_lcs = 0.0
                    scored_clauses.append((score, best_lcs, clause))
                scored_clauses.sort(key=lambda item: item[0], reverse=True)
                item_warnings: list[str] = []
                eligible = [
                    entry for entry in scored_clauses if entry[1] >= 0.5
                ]
                if eligible:
                    chosen_clauses = eligible[:2]
                    remaining = eligible[2:]
                    action_grams = _action_bigrams(phrases)
                    covered = any(
                        gram in _normalise(entry[2].body)
                        for entry in chosen_clauses
                        for gram in action_grams
                    )
                    if remaining and action_grams and not covered:
                        candidate_body = _normalise(remaining[0][2].body)
                        if any(gram in candidate_body for gram in action_grams):
                            chosen_clauses.append(remaining[0])
                elif scored_clauses:
                    chosen_clauses = [scored_clauses[0]]
                    item_warnings.append(
                        "课标中没有与该指标逐条对应的原文，"
                        f"已列最相近的“{scored_clauses[0][2].body[:40]}…”，"
                        "需人工确认是否适用。"
                    )
                else:
                    chosen_clauses = []
                    item_warnings.append(
                        "未能在所选课标范围内取得要求原文，需人工补充。"
                    )
                requirements: list[OutlineRequirement] = []
                for score, _, clause in chosen_clauses:
                    evidence = self._evidence_from_clause(clause, scope)
                    requirements.append(
                        OutlineRequirement(
                            clause=self._display_marker(clause.marker),
                            verb=clause.verb,
                            level=LEVEL_LABELS.get(clause.level, clause.level),
                            text=clause.body,
                            evidence=evidence,
                            verified=evidence.verified,
                            note=(
                                f"自动匹配度 {score:.2f}"
                                + ("；匹配度偏低，需人工确认" if score < 0.2 else "")
                            ),
                        )
                    )
                section_candidates: list[tuple[float, SectionRecord]] = []
                item_is_application = (
                    self._item_type(content) == "application"
                )
                action_chars = set("列解求检代化去识移算证画判消除")
                action_phrases = [
                    phrase
                    for phrase in phrases
                    if any(character in action_chars for character in phrase)
                ] or list(phrases)
                for record in theme_sections.get(theme["theme_code"], []):
                    blob = f"{record.title} {record.chapter} {record.text[:2500]}"
                    score = _section_scores(
                        query,
                        phrases,
                        blob,
                        section_table,
                        section_default,
                    )
                    title_score = max(
                        (
                            _lcs_ratio(phrase, record.title)
                            for phrase in action_phrases
                        ),
                        default=0.0,
                    )
                    substring = max(
                        (
                            _common_substring_ratio(phrase, blob)
                            for phrase in action_phrases
                        ),
                        default=0.0,
                    )
                    combined = 0.5 * score + 0.2 * title_score + 0.3 * substring
                    section_is_application = (
                        "实际问题" in record.title or "应用" in record.title
                    )
                    if item_is_application:
                        combined += 0.08 if section_is_application else -0.04
                    elif section_is_application:
                        combined -= 0.12
                    if _qualifier_conflict(
                        content,
                        record.title,
                    ):
                        combined *= 0.35
                    section_candidates.append((combined, record))
                section_candidates.sort(key=lambda item: item[0], reverse=True)
                chosen_sections: list[tuple[float, SectionRecord]] = []
                if section_candidates:
                    best_section = section_candidates[0][0]
                    chosen_sections = [
                        (score, record)
                        for score, record in section_candidates
                        if score >= max(0.16, best_section * 0.5)
                    ][:2]
                    if (
                        chosen_sections
                        and chosen_sections[0][1].level == 2
                        and not item_is_application
                    ):
                        best_record = chosen_sections[0][1]
                        siblings = [
                            record
                            for record in theme_sections.get(
                                theme["theme_code"],
                                [],
                            )
                            if record.document_id == best_record.document_id
                            and record.chapter == best_record.chapter
                            and record.level == 2
                        ]
                        siblings.sort(key=lambda record: record.page_start)
                        position = next(
                            (
                                index
                                for index, record in enumerate(siblings)
                                if record.id == best_record.id
                            ),
                            -1,
                        )
                        neighbor = None
                        if position > 0:
                            neighbor = siblings[position - 1]
                        elif 0 <= position < len(siblings) - 1:
                            neighbor = siblings[position + 1]
                        if neighbor is not None:
                            score_by_id = {
                                record.id: score
                                for score, record in section_candidates
                            }
                            chosen_sections = [
                                (
                                    score_by_id.get(best_record.id, 1.0),
                                    best_record,
                                ),
                                (
                                    score_by_id.get(neighbor.id, 0.0),
                                    neighbor,
                                ),
                            ]
                    if not chosen_sections:
                        chosen_sections = section_candidates[:1]
                else:
                    item_warnings.append(
                        "所选教材切片中没有与本内容对应的小节，材料出处不完整。"
                    )
                sources: list[OutlineEvidence] = []
                key_points: list[str] = []
                for _, record in chosen_sections:
                    phrase = max(
                        phrases,
                        key=lambda value: _lcs_ratio(value, record.title),
                    )
                    sources.append(self._section_evidence(record, scope, phrase))
                    for title in self._key_points(record, scope):
                        if title not in key_points:
                            key_points.append(title)
                    for _, evidence in self._unit_evidence(
                        record,
                        scope,
                        phrases,
                    )[:2]:
                        if all(
                            source.unit_id != evidence.unit_id
                            for source in sources
                        ):
                            sources.append(evidence)
                        unit_title = next(
                            (
                                unit["title"]
                                for unit in self.repo.list_units(
                                    document_id=record.document_id
                                )
                                if unit["id"] == evidence.unit_id
                            ),
                            "",
                        )
                        if unit_title and unit_title not in key_points:
                            key_points.append(unit_title)
                if not key_points:
                    key_points = [
                        section.title for _, section in chosen_sections
                    ]
                if not requirements:
                    item_warnings.append("缺少对应课标要求。")
                if not sources:
                    item_warnings.append("缺少教材材料出处。")
                items.append(
                    OutlineItem(
                        item_key=indicator["indicator_code"],
                        ordinal=len(items) + 1,
                        theme_code=theme["theme_code"],
                        theme_name=theme["theme_name"],
                        item_type=self._item_type(content),
                        content=content,
                        key_points=key_points[:8],
                        indicator_codes=[indicator["indicator_code"]],
                        requirements=requirements,
                        sources=sources[:4],
                        notes=(
                            f"由课程要求指标 {indicator['indicator_code']} 与"
                            "课程标准条目、教材小节按术语重合和章节顺序自动对应；"
                            "需教师确认后作为学情分析与学习建议的基准。"
                        ),
                        review_warnings=item_warnings,
                        status=(
                            OutlineItemStatus.NEEDS_REVIEW.value
                            if item_warnings
                            else OutlineItemStatus.DRAFT.value
                        ),
                    )
                )
        if not items:
            raise ValueError("没有生成任何学习内容，请检查课程要求。")
        return items

    @staticmethod
    def _display_marker(marker: str) -> str:
        return marker.strip()

    def _relation_evidence(
        self,
        record: SectionRecord,
        scope: dict[str, Any],
    ) -> OutlineEvidence:
        return OutlineEvidence(
            kind=OutlineSourceKind.TEXTBOOK.value,
            material_set_id=scope["textbook"]["material_set_id"],
            material_name=scope["textbook"]["material_name"],
            document_id=record.document_id,
            file_name=record.file_name,
            section_id=record.id,
            section_path=record.path,
            page=record.page_start,
            page_end=record.page_end,
            quote=record.title,
            verified=True,
            verification_note="章节标题与页码范围来自切分结果，用于核对编排顺序",
        )

    def rule_relations(
        self,
        scope: dict[str, Any],
        items: Sequence[OutlineItem],
    ) -> list[OutlineRelation]:
        records = self._scope_section_records(scope)
        relations: list[OutlineRelation] = []

        def add(
            source: OutlineItem,
            target: OutlineItem,
            relation_type: str,
            note: str,
            basis: str,
            evidence: Sequence[OutlineEvidence],
            status: str = OutlineItemStatus.DRAFT.value,
        ) -> None:
            if any(
                existing.source_item_key == source.item_key
                and existing.target_item_key == target.item_key
                for existing in relations
            ):
                return
            relations.append(
                OutlineRelation(
                    source_item_key=source.item_key,
                    target_item_key=target.item_key,
                    relation_type=relation_type,
                    note=note,
                    basis=basis,
                    evidence=list(evidence),
                    origin="rule",
                    status=status,
                )
            )

        def order_note(
            source: OutlineItem,
            target: OutlineItem,
            source_record: SectionRecord,
            target_record: SectionRecord,
        ) -> str:
            if source_record.file_name != target_record.file_name:
                return (
                    f"教材“{source_record.title}”（{source_record.file_name} 第 "
                    f"{source_record.page_start}–{source_record.page_end} 页）"
                    f"先于“{target_record.title}”（{target_record.file_name} 第 "
                    f"{target_record.page_start}–{target_record.page_end} 页）出现；"
                    f"“{target.content}”在“{source.content}”之后展开。"
                )
            if source_record.page_end <= target_record.page_start:
                return (
                    f"教材《{source_record.file_name}》先安排"
                    f"“{source_record.title}”（第 {source_record.page_start}–"
                    f"{source_record.page_end} 页），后安排“{target_record.title}”"
                    f"（第 {target_record.page_start}–{target_record.page_end} 页）；"
                    f"“{target.content}”在“{source.content}”之后展开，"
                    "按教材编排作为学习顺序依据。"
                )
            return (
                f"教材在同一小节群中安排“{source_record.title}”"
                f"（第 {source_record.page_start}–{source_record.page_end} 页）与"
                f"“{target_record.title}”（第 {target_record.page_start}–"
                f"{target_record.page_end} 页）；"
                f"“{target.content}”需要以“{source.content}”为基础。"
            )

        def primary_section(item: OutlineItem) -> SectionRecord | None:
            for source in item.sources:
                record = records.get(source.section_id or "")
                if record is not None:
                    return record
            return None

        def sections_of(item: OutlineItem) -> list[SectionRecord]:
            selected: list[SectionRecord] = []
            for source in item.sources:
                record = records.get(source.section_id or "")
                if record is not None and record not in selected:
                    selected.append(record)
            return selected

        # 1) application links: each application target points to the closest
        #    preceding content in the same theme.
        for target in items:
            if not any(marker in target.content for marker in _APPLICATION_MARKERS):
                continue
            for target_record in sections_of(target):
                candidates: list[tuple[int, OutlineItem, SectionRecord]] = []
                for source in items:
                    if (
                        source.item_key == target.item_key
                        or source.theme_code != target.theme_code
                    ):
                        continue
                    source_record = primary_section(source)
                    if source_record is None:
                        continue
                    if source_record.chapter != target_record.chapter:
                        continue
                    if source_record.page_end <= target_record.page_start:
                        candidates.append(
                            (source_record.page_end, source, source_record)
                        )
                if not candidates:
                    continue
                candidates.sort(key=lambda entry: entry[0], reverse=True)
                _, source, source_record = candidates[0]
                note = (
                    f"“{target.content}”需要以“{source.content}”为工具或基础："
                    + order_note(source, target, source_record, target_record)
                    + "属于同一主题内的应用与延伸。"
                )
                add(
                    source,
                    target,
                    OutlineRelationType.APPLICATION.value,
                    note,
                    "依据教材小节先后顺序与应用性表述。",
                    [
                        self._relation_evidence(source_record, scope),
                        self._relation_evidence(target_record, scope),
                    ],
                )

        # 2) sequential prerequisite links inside each theme.
        for theme in scope["themes"]:
            theme_items = [
                item for item in items if item.theme_code == theme["theme_code"]
            ]
            for index in range(1, len(theme_items)):
                source = theme_items[index - 1]
                target = theme_items[index]
                if any(
                    marker in target.content for marker in _APPLICATION_MARKERS
                ) and any(
                    relation.target_item_key == target.item_key
                    for relation in relations
                ):
                    continue
                source_record = primary_section(source)
                target_record = primary_section(target)
                if source_record is None or target_record is None:
                    continue
                target_sections = sections_of(target)
                advanced = max(
                    target_sections,
                    key=lambda record: record.page_start,
                )
                prior = [
                    record
                    for record in sections_of(source)
                    if record.page_end <= advanced.page_start
                ]
                if prior:
                    source_record = max(
                        prior,
                        key=lambda record: record.page_end,
                    )
                    target_record = advanced
                same_chapter = (
                    source_record.document_id == target_record.document_id
                    and source_record.chapter == target_record.chapter
                )
                if (
                    not same_chapter
                    and source_record.page_end > target_record.page_start
                ):
                    continue
                note = order_note(source, target, source_record, target_record)
                add(
                    source,
                    target,
                    OutlineRelationType.PREREQUISITE.value,
                    (
                        f"教材“{source_record.title}”与“{target_record.title}”"
                        "的先后安排显示后者依赖前者；"
                        + note
                        + "后续内容的解法与表述依赖前一内容。"
                    ),
                    "依据教材章节先后顺序（切分结果），非原文直接表述。",
                    [
                        self._relation_evidence(source_record, scope),
                        self._relation_evidence(target_record, scope),
                    ],
                )

        # 3) contrast links between equation / inequality contents.
        for index, left in enumerate(items):
            for right in items[index + 1 :]:
                if left.theme_code == right.theme_code:
                    continue
                left_has = "不等" in left.content
                right_has = "不等" in right.content
                if left_has == right_has:
                    continue
                left_text = left.content + left.theme_name
                right_text = right.content + right.theme_name
                similarity = min(
                    _lcs_ratio(left_text, right_text),
                    _lcs_ratio(right_text, left_text),
                )
                if similarity < 0.35:
                    continue
                equation_item = left if not left_has else right
                inequality_item = right if equation_item is left else left
                evidence: list[OutlineEvidence] = [
                    requirement.evidence
                    for requirement in [
                        *equation_item.requirements[:1],
                        *inequality_item.requirements[:1],
                    ]
                    if requirement.evidence is not None
                ]
                add(
                    equation_item,
                    inequality_item,
                    OutlineRelationType.CONTRAST.value,
                    (
                        f"“{equation_item.content}”与“{inequality_item.content}”"
                        "的对象与依据结构相近，但等式与不等式在两边乘（或除以）"
                        "负数时是否改变符号方向上存在关键差异，宜成组对比复习。"
                    ),
                    "依据两项指标的名称/要求文本相似度与“等式/不等式”差异自动识别。",
                    evidence,
                    status=OutlineItemStatus.NEEDS_REVIEW.value,
                )

        # 4) cross-theme link for contents built on "数量关系".
        for target in items:
            if "列" not in target.content:
                continue
            standard_text = " ".join(
                requirement.text for requirement in target.requirements
            )
            if "数量关系" not in standard_text:
                continue
            for source in items:
                if (
                    source.item_key == target.item_key
                    or "数量关系" not in source.content
                ):
                    continue
                source_record = primary_section(source)
                target_record = primary_section(target)
                if source_record is None or target_record is None:
                    continue
                add(
                    source,
                    target,
                    OutlineRelationType.PREREQUISITE.value,
                    (
                        f"课标在“{target.content}”的条目中要求根据“数量关系”列式；"
                        f"“{source.content}”正是用代数式表示数量关系的直接准备。"
                    ),
                    "依据课标条目中出现“数量关系”及教材先行章节。",
                    [
                        self._relation_evidence(source_record, scope),
                        *[
                            requirement.evidence
                            for requirement in target.requirements[:1]
                            if requirement.evidence is not None
                        ],
                    ],
                    status=OutlineItemStatus.NEEDS_REVIEW.value,
                )
        return relations

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate_items(
        self,
        scope: dict[str, Any],
        items: Sequence[OutlineItem],
    ) -> list[OutlineItem]:
        standard_pages = self.repo.list_pages(scope["standard"]["document_id"])
        standard_text = {
            int(page["page_number"]): page["final_text"] for page in standard_pages
        }
        page_texts: dict[str, dict[int, str]] = {}
        for entry in scope["textbook"]["sections"]:
            document_id = entry["document_id"]
            if document_id not in page_texts:
                page_texts[document_id] = {
                    int(page["page_number"]): page["final_text"]
                    for page in self.repo.list_pages(document_id)
                }
        section_by_id = {
            entry["section_id"]: entry for entry in scope["textbook"]["sections"]
        }
        valid_indicators = {
            indicator["indicator_code"]
            for theme in scope["themes"]
            for indicator in theme["indicators"]
        }
        checked: list[OutlineItem] = []
        for item in items:
            warnings = list(item.review_warnings)
            requirements: list[OutlineRequirement] = []
            for requirement in item.requirements:
                evidence = requirement.evidence
                if evidence is None:
                    warnings.append("要求缺少材料出处。")
                    requirements.append(requirement)
                    continue
                if evidence.kind == OutlineSourceKind.STANDARD.value:
                    page_text = standard_text.get(int(evidence.page or 0), "")
                    if not _quote_on_page(evidence.quote, page_text):
                        warnings.append(
                            f"课标第 {evidence.page} 页未找到要求引文，需人工核对。"
                        )
                        evidence = evidence.model_copy(
                            update={
                                "verified": False,
                                "verification_note": "引文未能在所引页逐字匹配",
                            }
                        )
                    else:
                        evidence = evidence.model_copy(
                            update={
                                "verified": True,
                                "verification_note": "引文已回查课标该页文本",
                            }
                        )
                    requirement = requirement.model_copy(
                        update={
                            "evidence": evidence,
                            "verified": bool(evidence.verified),
                        }
                    )
                requirements.append(requirement)
            sources: list[OutlineEvidence] = []
            for source in item.sources:
                if source.kind == OutlineSourceKind.STANDARD.value:
                    page_text = standard_text.get(int(source.page or 0), "")
                    verified = _quote_on_page(source.quote, page_text)
                    note = (
                        "引文已回查课标该页文本"
                        if verified
                        else "引文未能在所引页逐字匹配"
                    )
                    sources.append(
                        source.model_copy(
                            update={"verified": verified, "verification_note": note}
                        )
                    )
                    if not verified:
                        warnings.append(
                            f"课标第 {source.page} 页出处引文未通过核对。"
                        )
                    continue
                section = section_by_id.get(source.section_id or "")
                if section is None:
                    warnings.append("教材出处小节不在本次范围内。")
                    sources.append(source.model_copy(update={"verified": False}))
                    continue
                texts = page_texts.get(source.document_id, {})
                page = int(source.page or section["page_start"])
                if not (section["page_start"] <= page <= section["page_end"]):
                    warnings.append(
                        f"教材第 {page} 页不在小节“{section['title']}”范围内。"
                    )
                    sources.append(
                        source.model_copy(
                            update={
                                "verified": False,
                                "verification_note": "页码超出小节范围",
                            }
                        )
                    )
                    continue
                verified = _quote_on_page(source.quote, texts.get(page, ""))
                sources.append(
                    source.model_copy(
                        update={
                            "verified": verified,
                            "verification_note": (
                                "引文已回查教材该页文本"
                                if verified
                                else "引文未能在该页逐字匹配"
                            ),
                        }
                    )
                )
                if not verified:
                    warnings.append(
                        f"教材第 {page} 页出处引文未通过核对。"
                    )
            if not item.requirements:
                warnings.append("缺少对应课程标准要求。")
            if not item.sources:
                warnings.append("缺少教材材料出处。")
            unknown = [
                code
                for code in item.indicator_codes
                if code not in valid_indicators
            ]
            if unknown:
                warnings.append("指标编号不在本次范围：" + "、".join(unknown))
            if not _normalise(item.content):
                warnings.append("学习内容为空。")
            status = (
                OutlineItemStatus.NEEDS_REVIEW.value
                if warnings
                else item.status
            )
            if not warnings and item.status == OutlineItemStatus.NEEDS_REVIEW.value:
                status = OutlineItemStatus.DRAFT.value
            checked.append(
                item.model_copy(
                    update={
                        "requirements": requirements,
                        "sources": sources,
                        "review_warnings": list(dict.fromkeys(warnings)),
                        "status": status,
                    }
                )
            )
        return checked

    # ------------------------------------------------------------------
    # AI assistance
    # ------------------------------------------------------------------
    def prepare_two_stage_scope(
        self,
        *,
        mode: str,
        standard_material_id: str,
        standard_document_id: str,
        textbook_material_ids: Sequence[str],
        textbook_document_ids: Sequence[str] = (),
        dataset_id: str | None = None,
        theme_codes: Sequence[str] = (),
    ) -> dict[str, Any]:
        if mode not in SCOPE_MODE_LABELS:
            raise ValueError("未知范围模式")
        standard_material = self.repo.get_material_set(standard_material_id)
        standard_document = self.repo.get_document(standard_document_id)
        if (
            standard_material is None
            or standard_document is None
            or standard_document["material_set_id"] != standard_material_id
        ):
            raise ValueError("找不到所选课程标准版本")
        if not textbook_material_ids:
            raise ValueError("请至少选择一本课本")

        standard_sections = [
            {
                **section,
                "material_set_id": standard_material_id,
                "material_name": standard_material["name"],
                "file_name": standard_document["file_name"],
                "document_id": standard_document_id,
            }
            for section in self.repo.list_sections(standard_document_id)
        ]
        textbook_materials: list[dict[str, Any]] = []
        textbook_directory: list[dict[str, Any]] = []
        selected_document_ids = set(textbook_document_ids)
        for material_id in textbook_material_ids:
            material = self.repo.get_material_set(material_id)
            if material is None or material["material_type"] != "textbook":
                raise ValueError("所选课本材料不存在")
            textbook_materials.append(
                {
                    "id": material["id"],
                    "name": material["name"],
                    "version": material["version"],
                }
            )
            for document in self.repo.list_documents(material_id):
                if mode == "textbook" and document["id"] not in selected_document_ids:
                    continue
                for section in self.repo.list_sections(document["id"]):
                    textbook_directory.append(
                        {
                            "section_id": section["id"],
                            "document_id": document["id"],
                            "material_set_id": material_id,
                            "material_name": material["name"],
                            "file_name": document["file_name"],
                            "title": section["title"],
                            "path": section["path"],
                            "level": section["level"],
                            "page_start": section["page_start"],
                            "page_end": section["page_end"],
                        }
                    )
        if mode == "textbook" and not textbook_directory:
            raise ValueError("请至少选择一个课本切片")

        themes: list[dict[str, Any]] = []
        dataset: dict[str, Any] | None = None
        if mode == "theme":
            if not dataset_id:
                raise ValueError("主题模式需要选择学情数据集")
            dataset = self.get_dataset(dataset_id)
            if dataset is None:
                raise ValueError("找不到学情数据集")
            themes = [
                theme
                for theme in self.list_themes(dataset_id)
                if not theme_codes or theme["theme_code"] in theme_codes
            ]
            if not theme_codes:
                raise ValueError("请至少选择一个主题")
            if not themes:
                raise ValueError("所选主题不存在")

        standard_directory = [
            {
                "section_id": section["id"],
                "title": section["title"],
                "path": section["path"],
                "level": section["level"],
                "page_start": section["page_start"],
                "page_end": section["page_end"],
            }
            for section in standard_sections
        ]
        scope = {
            "mode": mode,
            "mode_label": SCOPE_MODE_LABELS[mode],
            "dataset": dataset,
            "themes": themes,
            "standard": {
                "material_set_id": standard_material["id"],
                "material_name": standard_material["name"],
                "version": standard_material["version"],
                "document_id": standard_document["id"],
                "file_name": standard_document["file_name"],
                "directory": standard_directory,
            },
            "textbook": {
                "materials": textbook_materials,
                "directory": textbook_directory,
            },
            "boundary": {
                "statement": (
                    f"{SCOPE_MODE_LABELS[mode]}；课程标准"
                    f"《{standard_material['name']}》第 {standard_material['version']} 版；"
                    f"课本 {len(textbook_materials)} 本、目录项 "
                    f"{len(textbook_directory)} 个"
                    + (
                        "；学情数据集"
                        f"“{dataset['name']}”；主题 "
                        + "、".join(
                            f"{theme['theme_code']} {theme['theme_name']}"
                            for theme in themes
                        )
                        if dataset is not None
                        else "；不涉及学情数据"
                    )
                )
            },
            "created_at": utc_now(),
        }
        return scope

    def _selection_prompt(self, scope: dict[str, Any]) -> str:
        payload = {
            "模式": scope["mode_label"],
            "学情数据": (
                {
                    "数据集": scope["dataset"]["name"],
                    "主题": scope["themes"],
                }
                if scope.get("dataset")
                else "不涉及学情数据"
            ),
            "课程标准": {
                "材料": scope["standard"]["material_name"],
                "版本": scope["standard"]["version"],
                "切片": scope["standard"]["file_name"],
                "目录": scope["standard"]["directory"],
            },
            "课本": {
                "材料": scope["textbook"]["materials"],
                "目录": scope["textbook"]["directory"],
            },
        }
        return f"""
你是课程材料检索助手。请根据当前范围和目录，只选择后续生成“学习内容与要求清单”
真正需要的课程标准目录项与课本目录项。不要返回正文，不要补充目录之外的 ID。

选择依据：
- 课本模式：以所选课本目录为主，结合课程标准目录确定需要读取的章节。
- 主题模式：以所选主题、主题目标和观察指标为主，在所选课本的全部目录中检索。
- 不要发送具体要求，也不要生成学习内容。

输入：
{json.dumps(payload, ensure_ascii=False, indent=2)}

只输出合法 JSON，不要 Markdown：
{{
  "standard_section_ids": ["课标目录 section_id"],
  "textbook_section_ids": ["课本目录 section_id"],
  "reasons": ["简短选择理由"]
}}
""".strip()

    def select_materials(
        self,
        *,
        scope: dict[str, Any],
        profile: ProviderProfile,
    ) -> dict[str, Any]:
        response, raw, _ = OpenAICompatibleClient(
            profile
        ).select_outline_sections(prompt=self._selection_prompt(scope))
        standard_ids = {
            item["section_id"] for item in scope["standard"]["directory"]
        }
        textbook_ids = {
            item["section_id"] for item in scope["textbook"]["directory"]
        }
        selected_standard = [
            item for item in response.standard_section_ids if item in standard_ids
        ]
        selected_textbook = [
            item for item in response.textbook_section_ids if item in textbook_ids
        ]
        warnings = []
        unknown_standard = len(selected_standard) != len(
            response.standard_section_ids
        )
        unknown_textbook = len(selected_textbook) != len(
            response.textbook_section_ids
        )
        if unknown_standard:
            warnings.append("目录检索阶段返回了范围外的课程标准目录项，已忽略。")
        if unknown_textbook:
            warnings.append("目录检索阶段返回了范围外的课本目录项，已忽略。")
        if not selected_standard and not selected_textbook:
            raise ValueError("目录检索阶段未返回有效目录项")
        return {
            "standard_section_ids": list(dict.fromkeys(selected_standard)),
            "textbook_section_ids": list(dict.fromkeys(selected_textbook)),
            "reasons": response.reasons,
            "warnings": warnings,
            "raw_response": raw,
        }

    def _generation_prompt(
        self,
        *,
        scope: dict[str, Any],
        selection: dict[str, Any],
        specific_requirements: str,
    ) -> str:
        standard_by_id = {
            item["id"]: item for item in self.repo.list_sections(
                scope["standard"]["document_id"]
            )
        }
        standard_content = [
            {
                "section_id": section_id,
                "title": standard_by_id[section_id]["title"],
                "path": standard_by_id[section_id]["path"],
                "page_start": standard_by_id[section_id]["page_start"],
                "page_end": standard_by_id[section_id]["page_end"],
                "content": standard_by_id[section_id]["content"],
            }
            for section_id in selection["standard_section_ids"]
            if section_id in standard_by_id
        ]
        directory_by_id = {
            item["section_id"]: item
            for item in scope["textbook"]["directory"]
        }
        textbook_content = []
        for section_id in selection["textbook_section_ids"]:
            entry = directory_by_id.get(section_id)
            if entry is None:
                continue
            section = self.repo.get_section(section_id)
            if section is None:
                continue
            textbook_content.append(
                {
                    **entry,
                    "content": section["content"],
                }
            )
        payload = {
            "模式": scope["mode_label"],
            "学情数据": (
                {
                    "数据集": scope["dataset"]["name"],
                    "主题": scope["themes"],
                }
                if scope.get("dataset")
                else "不涉及学情数据"
            ),
            "课程标准内容": standard_content,
            "课本内容": textbook_content,
            "具体要求": specific_requirements.strip(),
        }
        return f"""
你是初中数学课程与教学专家。请依据给出的具体内容生成格式化的学习内容梳理结果。
只能引用输入中出现的课程标准页、课本章节和原文，不得补充输入之外的页码或引文。

输出 JSON 必须包含：
1. 学习内容清单；
2. 每条学习内容对应的要求；
3. 每条要求对应的课标来源；
4. 每条学习内容的课本来源；
5. 学习内容之间的联系。

每个 item 必须包含 item_key、theme_code、item_type、content、key_points、
indicator_codes、requirements、sources 和 review_notes。requirements 中每条包含
text、clause、level、standard_page、quote；sources 中每条包含 kind、
material_set_id、document_id、section_id、page、quote。relations 使用 item_key。
具体要求可以为空，但不得因此虚构事实。只输出合法 JSON，不要 Markdown。

输入：
{json.dumps(payload, ensure_ascii=False, indent=2)}

JSON 结构：
{{
  "scope_note": "范围说明",
  "items": [
    {{
      "item_key": "T01I1",
      "theme_code": "T01",
      "item_type": "knowledge_point",
      "content": "学习内容",
      "key_points": ["要点"],
      "indicator_codes": [],
      "requirements": [
        {{
          "text": "要求",
          "clause": "①",
          "level": "掌握",
          "standard_page": 63,
          "quote": "逐字原文"
        }}
      ],
      "sources": [
        {{
          "kind": "textbook",
          "material_set_id": "set_xxx",
          "document_id": "doc_xxx",
          "section_id": "doc_xxx_p004_s005",
          "page": 4,
          "quote": "逐字原文"
        }}
      ],
      "review_notes": []
    }}
  ],
  "relations": [
    {{
      "source_key": "T01I1",
      "target_key": "T01I2",
      "relation_type": "prerequisite",
      "note": "说明",
      "basis": "依据"
    }}
  ]
}}
""".strip()

    def generate_from_selection(
        self,
        *,
        scope: dict[str, Any],
        selection: dict[str, Any],
        specific_requirements: str,
        profile: ProviderProfile,
        name: str | None = None,
    ) -> dict[str, Any]:
        mode = scope["mode"]
        dataset_id = (
            scope["dataset"]["id"]
            if scope.get("dataset")
            else self._ensure_textbook_scope_dataset()
        )
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        method = (
            "ai_two_stage_textbook"
            if mode == "textbook"
            else "ai_two_stage_theme"
        )
        run_name = (
            name
            or f"{SCOPE_MODE_LABELS[mode]} · AI 两阶段梳理 · "
            f"{datetime.now().astimezone().strftime('%m-%d %H:%M')}"
        ).strip()
        run_root = self.outlines_dir / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        stage1_raw = selection.get("raw_response") or ""
        (run_root / "stage1_selection.txt").write_text(
            stage1_raw,
            encoding="utf-8",
        )
        standard_by_id = {
            item["id"]: item
            for item in self.repo.list_sections(
                scope["standard"]["document_id"]
            )
        }
        textbook_by_id = {
            item["section_id"]: item
            for item in scope["textbook"]["directory"]
        }
        selected_standard = [
            {
                **standard_by_id[section_id],
                "material_set_id": scope["standard"]["material_set_id"],
                "material_name": scope["standard"]["material_name"],
                "file_name": scope["standard"]["file_name"],
                "document_id": scope["standard"]["document_id"],
            }
            for section_id in selection["standard_section_ids"]
            if section_id in standard_by_id
        ]
        selected_textbook = []
        for section_id in selection["textbook_section_ids"]:
            entry = textbook_by_id.get(section_id)
            section = self.repo.get_section(section_id)
            if entry is None or section is None:
                continue
            selected_textbook.append({**entry, "content": section["content"]})
        scope = {
            **scope,
            "standard": {
                **scope["standard"],
                "sections": selected_standard,
            },
            "textbook": {
                **scope["textbook"],
                "sections": selected_textbook,
                "material_set_id": (
                    selected_textbook[0]["material_set_id"]
                    if selected_textbook
                    else (
                        scope["textbook"]["materials"][0]["id"]
                        if scope["textbook"]["materials"]
                        else ""
                    )
                ),
                "material_name": (
                    selected_textbook[0]["material_name"]
                    if selected_textbook
                    else " / ".join(
                        item["name"]
                        for item in scope["textbook"]["materials"]
                    )
                ),
            },
        }
        prompt = self._generation_prompt(
            scope=scope,
            selection=selection,
            specific_requirements=specific_requirements,
        )
        client = OpenAICompatibleClient(profile)
        draft, raw_response, _ = client.draft_outline(prompt=prompt)
        (run_root / "stage2_generation.txt").write_text(
            raw_response,
            encoding="utf-8",
        )
        items, relations = self.ai_items(scope, draft, profile)
        for ordinal, item in enumerate(items, start=1):
            item.id = f"{run_id}_{item.item_key}"
            item.run_id = run_id
            item.ordinal = ordinal
        items = self.validate_items(scope, items)
        key_to_item = {item.item_key: item for item in items}
        for relation in relations:
            source = key_to_item.get(relation.source_item_key)
            target = key_to_item.get(relation.target_item_key)
            if source is None or target is None:
                continue
            relation.source_item_id = source.id
            relation.target_item_id = target.id
            relation.run_id = run_id
        relations = [
            relation
            for relation in relations
            if relation.source_item_id and relation.target_item_id
        ]
        scope = {
            **scope,
            "dataset": (
                {
                    "id": scope["dataset"]["id"],
                    "name": scope["dataset"]["name"],
                    "description": scope["dataset"].get("description") or "",
                }
                if scope.get("dataset")
                else {
                    "id": dataset_id,
                    "name": "课本模式",
                    "description": "不涉及学情数据",
                }
            ),
            "selection": {
                "standard_section_ids": selection["standard_section_ids"],
                "textbook_section_ids": selection["textbook_section_ids"],
                "reasons": selection.get("reasons", []),
                "warnings": selection.get("warnings", []),
            },
            "specific_requirements": specific_requirements.strip(),
            "workflow": [
                {
                    "stage": "范围准备与目录提交",
                    "detail": scope["boundary"]["statement"],
                },
                {
                    "stage": "目录检索阶段",
                    "detail": (
                        "选定课程标准目录 "
                        f"{len(selection['standard_section_ids'])} 项、"
                        "课本目录 "
                        f"{len(selection['textbook_section_ids'])} 项"
                    ),
                },
                {
                    "stage": "内容提取阶段",
                    "detail": "已从本地读取目录检索阶段选中的具体内容",
                },
                {
                    "stage": "结构化生成阶段",
                    "detail": "已生成学习内容、要求、来源和联系",
                },
                {
                    "stage": "结果校验与保存",
                    "detail": f"已保存 {len(items)} 条学习内容",
                },
            ],
        }
        raw_path = str(
            (run_root / "stage2_generation.txt").relative_to(self.paths.db_dir)
        )
        self.repo.create_outline_run(
            run_id=run_id,
            dataset_id=dataset_id,
            name=run_name,
            method=method,
            scope=scope,
            boundary_note=scope["boundary"]["statement"],
            status=(
                OutlineItemStatus.NEEDS_REVIEW.value
                if any(item.review_warnings for item in items)
                else OutlineItemStatus.DRAFT.value
            ),
            ai_vendor=profile.vendor_name,
            ai_base_url=profile.base_url,
            ai_model=profile.model,
            raw_response_path=raw_path,
            notes="AI 两阶段梳理：先选目录，再读取正文并结构化输出。",
        )
        self.repo.save_outline_items(
            run_id=run_id,
            items=items,
            change_source="ai_two_stage",
        )
        self.repo.save_outline_relations(
            run_id=run_id,
            relations=relations,
            origin="ai",
        )
        self.repo.update_outline_run(
            run_id,
            item_count=len(items),
            relation_count=len(relations),
        )
        self.write_artifacts(run_id)
        run = self.repo.get_outline_run(run_id)
        assert run is not None
        return {"run": run, "items": items, "relations": relations}

    def _ensure_textbook_scope_dataset(self) -> str:
        dataset_id = "lds_outline_textbook_scope"
        now = utc_now()
        with self.repo.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO learning_datasets
                (id, name, description, created_at, updated_at, deleted_at)
                VALUES (?, '课本模式', '不涉及学情数据', ?, ?, ?)
                """,
                (dataset_id, now, now, now),
            )
        return dataset_id

    def scope_mode(self, run: dict[str, Any]) -> str:
        return str(self.scope_of(run).get("mode") or "")

    def cleanup_legacy_runs(self) -> int:
        removed = 0
        for run in self.list_runs():
            if self.scope_mode(run) in SCOPE_MODE_LABELS:
                continue
            run_root = self.outlines_dir / run["id"]
            if self.repo.delete_outline_run(run["id"]):
                shutil.rmtree(run_root, ignore_errors=True)
                removed += 1
        valid_ids = {run["id"] for run in self.list_runs()}
        if self.outlines_dir.exists():
            for directory in self.outlines_dir.iterdir():
                if directory.is_dir() and directory.name not in valid_ids:
                    shutil.rmtree(directory, ignore_errors=True)
        return removed

    def _ai_prompt(self, scope: dict[str, Any]) -> str:
        clauses = self._scope_standard_clauses(scope)
        requirements = []
        for theme in scope["themes"]:
            for indicator in theme["indicators"]:
                requirements.append(
                    {
                        "主题编号": theme["theme_code"],
                        "主题": theme["theme_name"],
                        "主题目标": theme["theme_goal"],
                        "指标编号": indicator["indicator_code"],
                        "指标": indicator["indicator_name"],
                        "课程要求标注的课标页码": indicator["standard_page"],
                        "课程要求标注的教材页码": indicator["textbook_page"],
                    }
                )
        standard_lines = []
        for theme in scope["themes"]:
            for clause in clauses.get(theme["theme_code"], []):
                standard_lines.append(
                    f"[{theme['theme_code']}] 第{clause.page}页"
                    f"（正文{clause.printed_page or '?'}）：{clause.text}"
                )
        section_lines = []
        for entry in scope["textbook"]["sections"]:
            section_lines.append(
                f"[{entry['theme_code']}] {entry['file_name']}｜{entry['chapter']}｜"
                f"{entry['title']}｜第{entry['page_start']}-{entry['page_end']}页｜"
                f"section_id={entry['section_id']}｜"
                f"document_id={entry['document_id']}"
            )
        payload = {
            "范围说明": scope["boundary"]["statement"],
            "未纳入说明": scope["boundary"]["excluded_note"],
            "课程要求": requirements,
            "课标条目": standard_lines,
            "教材小节": section_lines,
        }
        return f"""
你是初中数学课程与教学专家。请依据给定范围，整理“学习内容与要求清单”和
“必要联系”。只能使用输入中出现的课标条目、教材小节和指标编号，不得补充
输入之外的页码、引文或材料。

范围与证据：
{json.dumps(payload, ensure_ascii=False, indent=2)}

输出要求：
1. 每个指标至少对应一个 item，item_key 使用指标编号（如 T03I1）。
2. content 是学习内容名称，应简明、准确，不与指标名重复堆砌。
3. key_points 是支撑该内容的关键知识点、性质、方法，最多 6 条。
4. requirements 中每条给出 text、clause、level，并给出课标页码与原文引文；
   quote 必须逐字来自输入课标条目，不得改写。
5. sources 只能引用输入中的教材小节，必须带 material_set_id、document_id、
   section_id、page 和逐字引文 quote；没有把握时宁可不写。
6. relations 使用 item_key 引用，relation_type 只能取 prerequisite（先备）、
   application（应用）、parallel（并列）、contrast（对比）。
7. note 说明联系的依据；basis 写明课标条目或教材章节依据。
8. 信息不足时在 review_notes 中说明，不要编造。
只输出一个 JSON 对象，不要 Markdown 代码围栏，结构如下：
{{
  "scope_note": "对本次范围的简短说明",
  "items": [
    {{
      "item_key": "T03I1",
      "theme_code": "T03",
      "item_type": "skill",
      "content": "不等式的性质与一元一次不等式的解法",
      "key_points": ["不等式的性质1、2、3"],
      "indicator_codes": ["T03I1"],
      "requirements": [
        {{
          "text": "探索不等式的基本性质",
          "clause": "①",
          "level": "经历",
          "standard_page": 63,
          "quote": "①结合具体问题，了解不等式的意义，探索不等式的基本性质。"
        }}
      ],
      "sources": [
        {{
          "kind": "textbook",
          "material_set_id": "set_xxx",
          "document_id": "doc_xxx",
          "section_id": "doc_xxx_p004_s005",
          "page": 4,
          "quote": "不等式两边乘（或除以）同一个正数……"
        }}
      ],
      "review_notes": []
    }}
  ],
  "relations": [
    {{
      "source_key": "T03I1",
      "target_key": "T03I2",
      "relation_type": "prerequisite",
      "note": "说明",
      "basis": "课标第63页条目②；教材11.1→11.3"
    }}
  ]
}}
""".strip()

    def ai_items(
        self,
        scope: dict[str, Any],
        draft: OutlineDraftResponse,
        profile: ProviderProfile,
    ) -> tuple[list[OutlineItem], list[OutlineRelation]]:
        theme_by_code = {
            theme["theme_code"]: theme for theme in scope["themes"]
        }
        section_by_id = {
            entry["section_id"]: entry for entry in scope["textbook"]["sections"]
        }
        valid_indicators = {
            indicator["indicator_code"]: (theme, indicator)
            for theme in scope["themes"]
            for indicator in theme["indicators"]
        }
        items: list[OutlineItem] = []
        key_to_id: dict[str, str] = {}
        seen_keys: set[str] = set()
        for index, draft_item in enumerate(draft.items, start=1):
            item_key = draft_item.item_key.strip() or f"KP{index:02d}"
            if item_key in seen_keys:
                item_key = f"{item_key}-{index}"
            seen_keys.add(item_key)
            theme = theme_by_code.get(draft_item.theme_code)
            indicator_codes = [
                code
                for code in draft_item.indicator_codes
                if code in valid_indicators
            ]
            if theme is None and indicator_codes:
                theme = valid_indicators[indicator_codes[0]][0]
            theme_code = theme["theme_code"] if theme else draft_item.theme_code
            theme_name = theme["theme_name"] if theme else ""
            if theme is None:
                theme = next(iter(theme_by_code.values()), None)
                theme_code = theme["theme_code"] if theme else theme_code
                theme_name = theme["theme_name"] if theme else theme_name
            warnings: list[str] = []
            requirements: list[OutlineRequirement] = []
            for entry in draft_item.requirements:
                page = entry.standard_page
                evidence = None
                if page:
                    evidence = OutlineEvidence(
                        kind=OutlineSourceKind.STANDARD.value,
                        material_set_id=scope["standard"]["material_set_id"],
                        material_name=scope["standard"]["material_name"],
                        document_id=scope["standard"]["document_id"],
                        file_name=scope["standard"]["file_name"],
                        page=page,
                        printed_page=None,
                        quote=entry.quote or entry.text,
                        verified=False,
                    )
                requirements.append(
                    OutlineRequirement(
                        clause=entry.clause,
                        verb="",
                        level=entry.level,
                        text=entry.text,
                        evidence=evidence,
                        verified=False,
                    )
                )
            sources: list[OutlineEvidence] = []
            for entry in draft_item.sources:
                section = section_by_id.get(entry.section_id)
                if section is None:
                    warnings.append("AI 引用了范围外的教材小节，已保留待核对。")
                sources.append(
                    OutlineEvidence(
                        kind=OutlineSourceKind.TEXTBOOK.value,
                        material_set_id=(
                            section.get("material_set_id")
                            or scope["textbook"]["material_set_id"]
                            if section
                            else scope["textbook"]["material_set_id"]
                        ),
                        material_name=(
                            section.get("material_name")
                            or scope["textbook"]["material_name"]
                            if section
                            else scope["textbook"]["material_name"]
                        ),
                        document_id=section["document_id"] if section else entry.document_id,
                        file_name=section["file_name"] if section else "",
                        section_id=entry.section_id,
                        section_path=section["path"] if section else "",
                        page=entry.page,
                        page_end=section["page_end"] if section else None,
                        quote=entry.quote,
                        unit_id=entry.unit_id,
                        verified=False,
                    )
                )
            warnings.extend(draft_item.review_notes)
            item = OutlineItem(
                item_key=item_key,
                ordinal=index,
                theme_code=theme_code,
                theme_name=theme_name,
                item_type=draft_item.item_type or "knowledge_point",
                content=draft_item.content,
                key_points=[
                    point.strip()
                    for point in draft_item.key_points
                    if point.strip()
                ][:8],
                indicator_codes=indicator_codes,
                requirements=requirements,
                sources=sources,
                notes="由 AI 依据所选范围起草，需逐条核对出处。",
                review_warnings=warnings,
                status=OutlineItemStatus.NEEDS_REVIEW.value,
            )
            items.append(item)
            key_to_id[item_key] = item.item_key
        relations: list[OutlineRelation] = []
        for draft_relation in draft.relations:
            if (
                draft_relation.source_key not in key_to_id
                or draft_relation.target_key not in key_to_id
                or draft_relation.source_key == draft_relation.target_key
            ):
                continue
            if draft_relation.relation_type not in set(OutlineRelationType):
                continue
            relations.append(
                OutlineRelation(
                    source_item_key=draft_relation.source_key,
                    target_item_key=draft_relation.target_key,
                    relation_type=draft_relation.relation_type,
                    note=draft_relation.note,
                    basis=draft_relation.basis,
                    origin="ai",
                    status=OutlineItemStatus.NEEDS_REVIEW.value,
                )
            )
        return items, relations

    # ------------------------------------------------------------------
    # runs and persistence
    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        scope: dict[str, Any],
        method: str = "rule",
        profile: ProviderProfile | None = None,
        name: str | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        if method not in {"rule", "ai"}:
            raise ValueError("未知的梳理方式")
        if method == "ai" and profile is None:
            raise ValueError("AI 梳理需要先配置当前厂商")
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        dataset_id = scope["dataset"]["id"]
        default_name = (
            f"{scope['dataset']['name']} · {METHOD_LABELS[method]} · "
            f"{datetime.now().astimezone().strftime('%m-%d %H:%M')}"
        )
        run_name = (name or default_name).strip()
        raw_path: str | None = None
        if progress:
            progress(1, 4, "准备范围与证据")
        if method == "rule":
            items = self.rule_items(scope)
            if progress:
                progress(2, 4, "生成学习内容与要求")
            for ordinal, item in enumerate(items, start=1):
                item.id = f"{run_id}_{item.item_key}"
                item.run_id = run_id
                item.ordinal = ordinal
            relations = self.rule_relations(scope, items)
        else:
            assert profile is not None
            client = OpenAICompatibleClient(profile)
            prompt = self._ai_prompt(scope)
            if progress:
                progress(2, 4, "调用 AI 梳理学习内容")
            draft, raw_response, _ = client.draft_outline(prompt=prompt)
            items, relations = self.ai_items(scope, draft, profile)
            for ordinal, item in enumerate(items, start=1):
                item.id = f"{run_id}_{item.item_key}"
                item.run_id = run_id
                item.ordinal = ordinal
            run_root = self.outlines_dir / run_id
            run_root.mkdir(parents=True, exist_ok=True)
            raw_file = run_root / "ai_response.txt"
            raw_file.write_text(raw_response, encoding="utf-8")
            raw_path = str(raw_file.relative_to(self.paths.db_dir))
        if progress:
            progress(3, 4, "核对材料出处")
        items = self.validate_items(scope, items)
        key_to_item = {item.item_key: item for item in items}
        for relation in relations:
            source = key_to_item.get(relation.source_item_key)
            target = key_to_item.get(relation.target_item_key)
            if source is None or target is None:
                continue
            relation.source_item_id = source.id
            relation.target_item_id = target.id
            relation.run_id = run_id
        relations = [
            relation
            for relation in relations
            if relation.source_item_id and relation.target_item_id
        ]
        self.repo.create_outline_run(
            run_id=run_id,
            dataset_id=dataset_id,
            name=run_name,
            method=method,
            scope=scope,
            boundary_note=scope["boundary"]["statement"]
            + (
                " " + scope["boundary"]["excluded_note"]
                if scope["boundary"].get("excluded_note")
                else ""
            ),
            status=(
                OutlineItemStatus.NEEDS_REVIEW.value
                if any(item.review_warnings for item in items)
                else OutlineItemStatus.DRAFT.value
            ),
            ai_vendor=profile.vendor_name if profile else None,
            ai_base_url=profile.base_url if profile else None,
            ai_model=profile.model if profile else None,
            raw_response_path=raw_path,
            notes=(
                "规则梳理：按术语重合、课标条目顺序与教材章节顺序自动生成。"
                if method == "rule"
                else "AI 梳理：模型起草后逐条回查课标页与教材页。"
            ),
        )
        self.repo.save_outline_items(
            run_id=run_id,
            items=items,
            change_source=method,
        )
        self.repo.save_outline_relations(
            run_id=run_id,
            relations=relations,
            origin=method,
        )
        self.repo.update_outline_run(
            run_id,
            item_count=len(items),
            relation_count=len(relations),
        )
        self.write_artifacts(run_id)
        if progress:
            progress(4, 4, "已保存梳理结果")
        run = self.repo.get_outline_run(run_id)
        assert run is not None
        return {
            "run": run,
            "items": items,
            "relations": relations,
        }

    def list_runs(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        return self.repo.list_outline_runs(dataset_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.repo.get_outline_run(run_id)

    def scope_of(self, run: dict[str, Any]) -> dict[str, Any]:
        try:
            return json.loads(run.get("scope_json") or "{}")
        except json.JSONDecodeError:
            return {}

    def items(self, run_id: str) -> list[OutlineItem]:
        return [
            self._item_from_row(row)
            for row in self.repo.list_outline_items(run_id)
        ]

    def relations(self, run_id: str) -> list[OutlineRelation]:
        return [
            self._relation_from_row(row)
            for row in self.repo.list_outline_relations(run_id)
        ]

    def _item_from_row(self, row: dict[str, Any]) -> OutlineItem:
        def decode(value: Any, fallback: Any) -> Any:
            if value is None or value == "":
                return fallback
            try:
                return json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return fallback

        data = {
            "id": row["id"],
            "run_id": row["run_id"],
            "item_key": row["item_key"],
            "ordinal": row["ordinal"],
            "theme_code": row.get("theme_code") or "",
            "theme_name": row.get("theme_name") or "",
            "item_type": row["item_type"],
            "content": row["content"],
            "key_points": decode(row.get("key_points_json"), []),
            "indicator_codes": decode(row.get("indicator_codes_json"), []),
            "requirements": decode(row.get("requirements_json"), []),
            "sources": decode(row.get("sources_json"), []),
            "notes": row.get("notes") or "",
            "review_warnings": decode(row.get("review_warnings_json"), []),
            "status": row.get("status") or OutlineItemStatus.DRAFT.value,
            "current_version": row.get("current_version") or 1,
        }
        return OutlineItem.model_validate(data)

    def _relation_from_row(self, row: dict[str, Any]) -> OutlineRelation:
        try:
            evidence = json.loads(row.get("evidence_json") or "[]")
        except json.JSONDecodeError:
            evidence = []
        data = {
            "id": row["id"],
            "run_id": row["run_id"],
            "source_item_id": row["source_item_id"],
            "source_item_key": row.get("source_item_key") or "",
            "target_item_id": row["target_item_id"],
            "target_item_key": row.get("target_item_key") or "",
            "relation_type": row["relation_type"],
            "note": row.get("note") or "",
            "basis": row.get("basis") or "",
            "evidence": evidence,
            "origin": row.get("origin") or "rule",
            "status": row.get("status") or OutlineItemStatus.DRAFT.value,
            "created_at": row.get("created_at") or "",
        }
        return OutlineRelation.model_validate(data)

    def update_run_boundary(self, run_id: str, boundary_note: str) -> None:
        run = self.repo.get_outline_run(run_id)
        if run is None:
            raise ValueError("找不到该次梳理")
        scope = self.scope_of(run)
        boundary = scope.setdefault("boundary", {})
        boundary["statement"] = boundary_note.strip()
        self.repo.update_outline_run(
            run_id,
            boundary_note=boundary_note.strip(),
            scope=scope,
        )

    def add_manual_item(
        self,
        *,
        run_id: str,
        theme_code: str,
        content: str,
        key_points: Sequence[str],
        indicator_codes: Sequence[str],
        notes: str = "",
        section_id: str | None = None,
        page: int | None = None,
        quote: str = "",
        standard_quote: str = "",
        standard_page: int | None = None,
    ) -> OutlineItem:
        run = self.repo.get_outline_run(run_id)
        if run is None:
            raise ValueError("找不到该次梳理")
        scope = self.scope_of(run)
        theme = next(
            (
                item
                for item in scope.get("themes", [])
                if item["theme_code"] == theme_code
            ),
            None,
        )
        if theme is None:
            raise ValueError("找不到该主题")
        content = content.strip()
        if not content:
            raise ValueError("学习内容不能为空")
        existing = self.repo.list_outline_items(run_id)
        used_keys = {row["item_key"] for row in existing}
        index = 1
        while f"KP{index:02d}" in used_keys:
            index += 1
        warnings = ["人工添加，尚未经 AI 或规则核对。"]
        requirements: list[OutlineRequirement] = []
        if standard_quote.strip() and standard_page:
            evidence = OutlineEvidence(
                kind=OutlineSourceKind.STANDARD.value,
                material_set_id=scope["standard"]["material_set_id"],
                material_name=scope["standard"]["material_name"],
                document_id=scope["standard"]["document_id"],
                file_name=scope["standard"]["file_name"],
                page=int(standard_page),
                quote=standard_quote.strip(),
                verified=False,
            )
            requirements.append(
                OutlineRequirement(
                    text=standard_quote.strip(),
                    evidence=evidence,
                    verified=False,
                )
            )
        sources: list[OutlineEvidence] = []
        if section_id:
            section = next(
                (
                    entry
                    for entry in scope["textbook"]["sections"]
                    if entry["section_id"] == section_id
                ),
                None,
            )
            if section is None:
                raise ValueError("所选章节不在本次范围内")
            sources.append(
                OutlineEvidence(
                    kind=OutlineSourceKind.TEXTBOOK.value,
                    material_set_id=scope["textbook"]["material_set_id"],
                    material_name=scope["textbook"]["material_name"],
                    document_id=section["document_id"],
                    file_name=section["file_name"],
                    section_id=section_id,
                    section_path=section["path"],
                    page=int(page or section["page_start"]),
                    page_end=section["page_end"],
                    quote=quote.strip() or section["title"],
                    verified=False,
                )
            )
        item = OutlineItem(
            id=f"{run_id}_KP{index:02d}",
            run_id=run_id,
            item_key=f"KP{index:02d}",
            ordinal=len(existing) + 1,
            theme_code=theme_code,
            theme_name=theme["theme_name"],
            item_type=self._item_type(content),
            content=content,
            key_points=[point.strip() for point in key_points if point.strip()],
            indicator_codes=[code for code in indicator_codes if code],
            requirements=requirements,
            sources=sources,
            notes=notes.strip(),
            review_warnings=warnings,
            status=OutlineItemStatus.NEEDS_REVIEW.value,
        )
        items = self.validate_items(scope, [item])
        validated = items[0]
        self.repo.save_outline_items(
            run_id=run_id,
            items=[validated],
            change_source="manual",
        )
        self.repo.update_outline_run(
            run_id,
            item_count=len(existing) + 1,
        )
        self.write_artifacts(run_id)
        return validated

    def update_item(
        self,
        *,
        item_id: str,
        content: str,
        key_points: Sequence[str],
        indicator_codes: Sequence[str],
        notes: str,
        theme_code: str,
        theme_name: str,
        item_type: str,
        status: str | None = None,
    ) -> None:
        row = self.repo.get_outline_item(item_id)
        if row is None:
            raise ValueError("找不到该学习内容")
        self.repo.update_outline_item(
            item_id=item_id,
            content=content.strip(),
            key_points=[point.strip() for point in key_points if point.strip()],
            indicator_codes=[
                code.strip() for code in indicator_codes if code.strip()
            ],
            requirements=[
                OutlineRequirement.model_validate(entry)
                for entry in json.loads(row["requirements_json"] or "[]")
            ],
            sources=[
                OutlineEvidence.model_validate(entry)
                for entry in json.loads(row["sources_json"] or "[]")
            ],
            notes=notes.strip(),
            theme_code=theme_code,
            theme_name=theme_name,
            item_type=item_type,
            status=status,
        )

    def set_items_status(self, item_ids: Sequence[str], status: str) -> int:
        return self.repo.set_outline_items_status(item_ids, status)

    def delete_item(self, item_id: str) -> None:
        row = self.repo.get_outline_item(item_id)
        if row is None:
            raise ValueError("找不到该学习内容")
        self.repo.soft_delete_outline_items([item_id])
        run = self.repo.get_outline_run(row["run_id"])
        if run is not None:
            remaining = self.repo.list_outline_items(row["run_id"])
            self.repo.update_outline_run(
                row["run_id"],
                item_count=len(remaining),
            )
            self.write_artifacts(row["run_id"])

    def add_relation(
        self,
        *,
        run_id: str,
        source_item_id: str,
        target_item_id: str,
        relation_type: str,
        note: str,
        basis: str = "",
    ) -> None:
        self.repo.add_outline_relation(
            run_id=run_id,
            source_item_id=source_item_id,
            target_item_id=target_item_id,
            relation_type=relation_type,
            note=note,
            basis=basis,
        )
        run = self.repo.get_outline_run(run_id)
        if run is not None:
            self.repo.update_outline_run(
                run_id,
                relation_count=len(self.repo.list_outline_relations(run_id)),
            )
        self.write_artifacts(run_id)

    def delete_relation(self, relation_id: str) -> None:
        row = next(
            (
                relation
                for run in self.repo.list_outline_runs()
                for relation in self.repo.list_outline_relations(run["id"])
                if relation["id"] == relation_id
            ),
            None,
        )
        self.repo.delete_outline_relation(relation_id)
        if row is not None:
            self.repo.update_outline_run(
                row["run_id"],
                relation_count=len(
                    self.repo.list_outline_relations(row["run_id"])
                ),
            )
            self.write_artifacts(row["run_id"])

    # ------------------------------------------------------------------
    # exports
    # ------------------------------------------------------------------
    def references(
        self,
        items: Sequence[OutlineItem],
        relations: Sequence[OutlineRelation],
    ) -> list[dict[str, Any]]:
        used: dict[str, dict[str, Any]] = {}
        for item in items:
            for evidence in [
                *[requirement.evidence for requirement in item.requirements],
                *item.sources,
            ]:
                if evidence is None:
                    continue
                key = "|".join(
                    [
                        evidence.kind,
                        evidence.document_id,
                        str(evidence.page),
                        evidence.quote,
                    ]
                )
                entry = used.setdefault(
                    key,
                    {
                        "ref_id": f"R{len(used) + 1:03d}",
                        "kind": SOURCE_KIND_LABELS.get(evidence.kind, evidence.kind),
                        "material": evidence.material_name,
                        "file": evidence.file_name,
                        "section": evidence.section_path,
                        "page": evidence.page,
                        "printed_page": evidence.printed_page,
                        "quote": evidence.quote,
                        "verified": evidence.verified,
                        "verification_note": evidence.verification_note,
                        "used_by": [],
                    },
                )
                if evidence.unit_id:
                    entry.setdefault("unit_ids", [])
                    if evidence.unit_id not in entry["unit_ids"]:
                        entry["unit_ids"].append(evidence.unit_id)
                if item.item_key not in entry["used_by"]:
                    entry["used_by"].append(item.item_key)
        for relation in relations:
            for evidence in relation.evidence:
                key = "|".join(
                    [
                        evidence.kind,
                        evidence.document_id,
                        str(evidence.page),
                        evidence.quote,
                    ]
                )
                entry = used.setdefault(
                    key,
                    {
                        "ref_id": f"R{len(used) + 1:03d}",
                        "kind": SOURCE_KIND_LABELS.get(evidence.kind, evidence.kind),
                        "material": evidence.material_name,
                        "file": evidence.file_name,
                        "section": evidence.section_path,
                        "page": evidence.page,
                        "printed_page": evidence.printed_page,
                        "quote": evidence.quote,
                        "verified": evidence.verified,
                        "verification_note": evidence.verification_note,
                        "used_by": [],
                    },
                )
                relation_key = (
                    f"{relation.source_item_key}→{relation.target_item_key}"
                )
                if relation_key not in entry["used_by"]:
                    entry["used_by"].append(relation_key)
        return list(used.values())

    def bundle(self, run_id: str) -> dict[str, Any]:
        run = self.repo.get_outline_run(run_id)
        if run is None:
            raise ValueError("找不到该次梳理")
        items = self.items(run_id)
        relations = self.relations(run_id)
        return {
            "run": {
                "id": run["id"],
                "name": run["name"],
                "method": run["method"],
                "status": run["status"],
                "dataset_id": run["dataset_id"],
                "created_at": run["created_at"],
                "ai_vendor": run.get("ai_vendor"),
                "ai_model": run.get("ai_model"),
            },
            "scope": self.scope_of(run),
            "boundary_note": run["boundary_note"],
            "items": [item.model_dump() for item in items],
            "relations": [relation.model_dump() for relation in relations],
            "references": self.references(items, relations),
            "field_definitions": OUTLINE_FIELD_DEFINITIONS,
            "generated_at": utc_now(),
        }

    def write_artifacts(self, run_id: str) -> Path:
        bundle = self.bundle(run_id)
        root = self.outlines_dir / run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "outline.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / "scope.json").write_text(
            json.dumps(bundle["scope"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for name, rows in (
            ("items.jsonl", bundle["items"]),
            ("relations.jsonl", bundle["relations"]),
            ("references.jsonl", bundle["references"]),
        ):
            with (root / name).open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (root / "references.json").write_text(
            json.dumps(bundle["references"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / "items.csv").write_bytes(
            self._csv_bytes(
                ["编号", "主题编号", "主题", "类型", "学习内容", "要点", "指标编号",
                 "要求", "材料出处", "状态", "核对提示"],
                [
                    [
                        item["item_key"],
                        item["theme_code"],
                        item["theme_name"],
                        ITEM_TYPE_LABELS.get(item["item_type"], item["item_type"]),
                        item["content"],
                        "；".join(item["key_points"]),
                        "、".join(item["indicator_codes"]),
                        "；".join(
                            f"{requirement['level']}：{requirement['text']}"
                            for requirement in item["requirements"]
                        ),
                        "；".join(
                            f"{SOURCE_KIND_LABELS.get(source['kind'], source['kind'])}"
                            f"《{source['file_name']}》第{source['page']}页"
                            for source in item["sources"]
                        ),
                        STATUS_LABELS.get(item["status"], item["status"]),
                        "；".join(item["review_warnings"]),
                    ]
                    for item in bundle["items"]
                ],
            )
        )
        (root / "relations.csv").write_bytes(
            self._csv_bytes(
                ["源编号", "目标编号", "联系类型", "说明", "依据", "生成方式", "状态"],
                [
                    [
                        relation["source_item_key"],
                        relation["target_item_key"],
                        RELATION_TYPE_LABELS.get(
                            relation["relation_type"], relation["relation_type"]
                        ),
                        relation["note"],
                        relation["basis"],
                        relation["origin"],
                        STATUS_LABELS.get(relation["status"], relation["status"]),
                    ]
                    for relation in bundle["relations"]
                ],
            )
        )
        (root / "references.csv").write_bytes(
            self._csv_bytes(
                ["出处编号", "类型", "材料", "文件", "章节", "页码", "正文页", "原文",
                 "核对", "用于"],
                [
                    [
                        row["ref_id"],
                        row["kind"],
                        row["material"],
                        row["file"],
                        row["section"],
                        row["page"],
                        row["printed_page"],
                        row["quote"],
                        "通过" if row["verified"] else "未通过",
                        "、".join(row["used_by"]),
                    ]
                    for row in bundle["references"]
                ],
            )
        )
        (root / "field_definitions.csv").write_bytes(
            self._csv_bytes(
                list(OUTLINE_FIELD_DEFINITIONS[0].keys()),
                [list(row.values()) for row in OUTLINE_FIELD_DEFINITIONS],
            )
        )
        (root / "README.md").write_text(
            self._artifact_readme(bundle),
            encoding="utf-8",
        )
        return root

    @staticmethod
    def _csv_bytes(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> bytes:
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
        return output.getvalue().encode("utf-8-sig")

    def _artifact_readme(self, bundle: dict[str, Any]) -> str:
        return f"""# 学习内容梳理成果包

- 梳理编号：{bundle['run']['id']}
- 方式：{METHOD_LABELS.get(bundle['run']['method'], bundle['run']['method'])}
- 数据集：{bundle['run']['dataset_id']}
- 生成时间：{bundle['run']['created_at']}

## 范围说明

{bundle['boundary_note']}

## 文件

- `outline.json`：完整成果（范围、清单、联系、出处、字段说明）。
- `scope.json`：范围与纳入/未纳入章节。
- `items.jsonl` / `items.csv`：学习内容与要求清单。
- `relations.jsonl` / `relations.csv`：必要联系。
- `references.json` / `references.csv`：材料出处与核对结果。
- `field_definitions.csv`：字段说明。

## 核对约定

- 每条要求与教材出处都带原文引文；`verified` 表示引文能在该页提取文本中
  逐字匹配。未通过核对的项目保留在清单中，并在“核对提示”中标明。
- 信息不足时保留空值或“待复核”，不把缺失记录当作已掌握。
"""

    def export_bundle(self, run_id: str) -> bytes:
        bundle = self.bundle(run_id)
        return (
            json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")

    def export_items_csv(self, run_id: str) -> bytes:
        root = self.write_artifacts(run_id)
        return (root / "items.csv").read_bytes()

    def export_relations_csv(self, run_id: str) -> bytes:
        root = self.write_artifacts(run_id)
        return (root / "relations.csv").read_bytes()

    def export_references_csv(self, run_id: str) -> bytes:
        root = self.write_artifacts(run_id)
        return (root / "references.csv").read_bytes()

    def export_field_definitions_csv(self) -> bytes:
        return self._csv_bytes(
            list(OUTLINE_FIELD_DEFINITIONS[0].keys()),
            [list(row.values()) for row in OUTLINE_FIELD_DEFINITIONS],
        )


OUTLINE_FIELD_DEFINITIONS: list[dict[str, str]] = [
    {
        "数据表": "outline_runs",
        "字段": "id",
        "类型": "字符串",
        "含义": "一次梳理运行的唯一编号",
        "取值或格式": "run_ + 12 位十六进制",
        "示例": "run_1a2b3c4d5e6f",
        "为空时": "不允许为空",
    },
    {
        "数据表": "outline_runs",
        "字段": "method",
        "类型": "枚举",
        "含义": "清单生成方式",
        "取值或格式": "rule=规则梳理；ai=AI 梳理",
        "示例": "rule",
        "为空时": "不允许为空",
    },
    {
        "数据表": "outline_runs",
        "字段": "boundary_note",
        "类型": "长文本",
        "含义": "本次梳理的范围边界说明",
        "取值或格式": "自由文本，需写明已纳入与未纳入内容",
        "示例": "已确定主题范围，其余内容未纳入本次梳理",
        "为空时": "表示范围边界尚未说明",
    },
    {
        "数据表": "outline_items",
        "字段": "item_key",
        "类型": "字符串",
        "含义": "学习内容在本清单中的稳定编号",
        "取值或格式": "优先沿用课程要求指标编号，人工新增用 KP01 形式",
        "示例": "T03I1",
        "为空时": "不允许为空",
    },
    {
        "数据表": "outline_items",
        "字段": "theme_code",
        "类型": "字符串",
        "含义": "所属主题编号",
        "取值或格式": "与课程要求“主题编号”一致",
        "示例": "T03",
        "为空时": "表示未归入主题，需复核",
    },
    {
        "数据表": "outline_items",
        "字段": "content",
        "类型": "文本",
        "含义": "学习内容名称",
        "取值或格式": "简洁名词性表述，避免与要求混写",
        "示例": "不等式的性质与一元一次不等式的解法",
        "为空时": "不允许为空",
    },
    {
        "数据表": "outline_items",
        "字段": "key_points",
        "类型": "字符串数组",
        "含义": "支撑该学习内容的关键知识点、性质或方法",
        "取值或格式": "JSON 数组；每项须能追溯到教材小节或单元",
        "示例": "[\"不等式的性质1\", \"不等式的性质2\"]",
        "为空时": "表示尚未整理要点",
    },
    {
        "数据表": "outline_items",
        "字段": "indicator_codes",
        "类型": "字符串数组",
        "含义": "对应的课程要求观测指标编号",
        "取值或格式": "JSON 数组；取值须来自所选数据集",
        "示例": "[\"T03I1\"]",
        "为空时": "表示与课程要求尚未建立对应",
    },
    {
        "数据表": "outline_items",
        "字段": "requirements",
        "类型": "对象数组",
        "含义": "对应课程标准要求，含层级、原文与页码",
        "取值或格式": "对象含 text/clause/level/evidence",
        "示例": "{\"level\": \"掌握\", \"text\": \"能解数字系数的一元一次不等式\"}",
        "为空时": "表示未匹配到课标条目，需人工确认",
    },
    {
        "数据表": "outline_items",
        "字段": "sources",
        "类型": "对象数组",
        "含义": "教材材料出处，含文件、小节、页码与原文",
        "取值或格式": "对象含 document_id/section_id/page/quote",
        "示例": "{\"page\": 4, \"quote\": \"不等式的性质1……\"}",
        "为空时": "表示缺少教材出处",
    },
    {
        "数据表": "outline_items",
        "字段": "status",
        "类型": "枚举",
        "含义": "教师核对状态",
        "取值或格式": "draft=草稿；needs_review=待复核；verified=已确认",
        "示例": "needs_review",
        "为空时": "默认草稿",
    },
    {
        "数据表": "outline_items",
        "字段": "review_warnings",
        "类型": "字符串数组",
        "含义": "自动核对发现的问题，供教师处理",
        "取值或格式": "JSON 数组；问题描述",
        "示例": "[\"教材第 4 页出处引文未通过核对\"]",
        "为空时": "表示自动核对未发现问题",
    },
    {
        "数据表": "outline_relations",
        "字段": "source_item_key",
        "类型": "字符串",
        "含义": "联系的起点学习内容",
        "取值或格式": "须存在于同一 run 的 item_key",
        "示例": "T03I1",
        "为空时": "不允许为空",
    },
    {
        "数据表": "outline_relations",
        "字段": "target_item_key",
        "类型": "字符串",
        "含义": "联系的终点学习内容",
        "取值或格式": "须存在于同一 run 的 item_key，且不能与起点相同",
        "示例": "T03I2",
        "为空时": "不允许为空",
    },
    {
        "数据表": "outline_relations",
        "字段": "relation_type",
        "类型": "枚举",
        "含义": "联系类型",
        "取值或格式": "prerequisite=先备；application=应用；parallel=并列；contrast=对比；other=其他",
        "示例": "prerequisite",
        "为空时": "不允许为空",
    },
    {
        "数据表": "outline_relations",
        "字段": "note",
        "类型": "文本",
        "含义": "联系的具体说明",
        "取值或格式": "应写明前后内容及为什么有联系",
        "示例": "教材先安排 11.1，后安排 11.2；解法以性质为基础",
        "为空时": "不允许为空",
    },
    {
        "数据表": "outline_relations",
        "字段": "basis",
        "类型": "文本",
        "含义": "建立联系的依据",
        "取值或格式": "课标条目、教材章节或推断说明",
        "示例": "依据教材章节先后顺序",
        "为空时": "表示依据未说明",
    },
    {
        "数据表": "outline_relations",
        "字段": "origin",
        "类型": "枚举",
        "含义": "联系生成方式",
        "取值或格式": "rule=规则；ai=AI；manual=人工",
        "示例": "rule",
        "为空时": "默认 rule",
    },
    {
        "数据表": "references",
        "字段": "ref_id",
        "类型": "字符串",
        "含义": "出处编号",
        "取值或格式": "R + 3 位序号",
        "示例": "R001",
        "为空时": "不允许为空",
    },
    {
        "数据表": "references",
        "字段": "kind",
        "类型": "枚举",
        "含义": "出处类型",
        "取值或格式": "standard=课程标准；textbook=教材；requirement=课程要求",
        "示例": "standard",
        "为空时": "不允许为空",
    },
    {
        "数据表": "references",
        "字段": "page",
        "类型": "整数",
        "含义": "PDF 文件页号",
        "取值或格式": "从 1 开始",
        "示例": "63",
        "为空时": "表示页码未确认",
    },
    {
        "数据表": "references",
        "字段": "printed_page",
        "类型": "字符串",
        "含义": "页脚印刷页码（正文页）",
        "取值或格式": "数字字符串",
        "示例": "56",
        "为空时": "表示该页未识别到印刷页码",
    },
    {
        "数据表": "references",
        "字段": "quote",
        "类型": "文本",
        "含义": "用于核对的原文引文",
        "取值或格式": "逐字摘录，不概括改写",
        "示例": "②能解数字系数的一元一次不等式，并能在数轴上表示出解集；",
        "为空时": "表示尚未提供引文",
    },
    {
        "数据表": "references",
        "字段": "verified",
        "类型": "布尔",
        "含义": "引文是否在所指页面逐字匹配",
        "取值或格式": "true=通过；false=未通过",
        "示例": "true",
        "为空时": "默认 false，需人工核对",
    },
]
