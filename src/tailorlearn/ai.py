from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from tailorlearn.models import (
    AIResponse,
    AIResult,
    AIUnit,
    MaterialType,
    ProviderProfile,
)

_UNIT_TYPES = {
    MaterialType.TEXTBOOK: (
        "concept, property, theorem, example, exercise, activity, summary, other"
    ),
    MaterialType.STANDARD: (
        "content_requirement, academic_requirement, teaching_hint, "
        "core_competency, implementation_guidance, other"
    ),
}
_ALLOWED_UNIT_TYPES = {
    MaterialType.TEXTBOOK: {
        "concept",
        "property",
        "theorem",
        "example",
        "exercise",
        "activity",
        "summary",
        "other",
    },
    MaterialType.STANDARD: {
        "content_requirement",
        "academic_requirement",
        "teaching_hint",
        "core_competency",
        "implementation_guidance",
        "other",
    },
}


class AIRequestError(RuntimeError):
    pass


def _normalise_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("响应中没有 JSON 对象")
    return json.loads(cleaned[start : end + 1])


class OpenAICompatibleClient:
    def __init__(self, profile: ProviderProfile):
        if not profile.api_key:
            raise ValueError("当前厂商配置没有 API Key")
        self.profile = profile
        self.client = OpenAI(
            api_key=profile.api_key,
            base_url=profile.base_url,
            timeout=120,
            max_retries=1,
        )

    def test_connection(self) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.profile.model,
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with exactly: OK",
                    }
                ],
            )
        except Exception as exc:
            raise AIRequestError(f"连接失败：{exc}") from exc
        if not response.choices:
            raise AIRequestError("服务没有返回任何选项")
        return "连接成功"

    def _complete(self, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        try:
            response = self.client.chat.completions.create(
                model=self.profile.model,
                messages=messages,
            )
        except Exception as exc:
            raise AIRequestError(str(exc)) from exc
        if not response.choices:
            raise AIRequestError("AI 服务没有返回任何结果")
        choice = response.choices[0]
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise AIRequestError("AI 服务返回了空内容")
        return content, getattr(choice, "finish_reason", None)

    def _structure_prompt(
        self,
        *,
        material_type: str,
        section: dict[str, Any],
        content: str,
    ) -> str:
        resolved_type = MaterialType(material_type)
        types = _UNIT_TYPES[resolved_type]
        return f"""
你负责把教材切片或课程标准忠实整理成结构化 JSON。只整理给定的这一节，
不要补充材料中没有的信息，不要做摘要，不要做跨章节归纳，不要生成学习建议。

材料类型：{material_type}
章节路径：{section["path"]}
页码范围：{section["page_start"]}-{section["page_end"]}
允许的 unit_type：{types}

要求：
1. content 必须忠实保留条件、限定词、公式、要求层级。可修复明显断行和 OCR 噪声。
2. 每个单元给出关键原文摘录 source_refs，page 必须来自输入页码。
3. 例题保留题干、分析、解答；课程标准保留内容要求、学业要求、教学提示的区分。
4. 无法确认的内容不要补全。若原页文字疑似损坏，在 review_warnings 中说明。
5. related_unit_ids 固定返回空数组，导入阶段不建立跨材料关系。
6. 只输出一个 JSON 对象，不要 Markdown 代码围栏。

输出结构：
{{
  "units": [
    {{
      "unit_type": "concept",
      "title": "简短准确标题",
      "content": "忠实整理后的文本",
      "source_refs": [
        {{"page": 12, "quote": "用于核对的原文片段"}}
      ],
      "related_unit_ids": [],
      "review_warnings": []
    }}
  ]
}}

待整理内容：
{content}
""".strip()

    def _repair_json(self, broken: str) -> str:
        content, _ = self._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "修复下面的 JSON。只修正语法，不改变事实内容；"
                        "只输出修复后的 JSON 对象。"
                    ),
                },
                {"role": "user", "content": broken},
            ]
        )
        return content

    def _validate_units(
        self,
        units: Sequence[AIUnit],
        *,
        material_type: str,
        section: dict[str, Any],
        page_texts: dict[int, str],
    ) -> list[AIUnit]:
        checked: list[AIUnit] = []
        allowed_types = _ALLOWED_UNIT_TYPES[MaterialType(material_type)]
        start = int(section["page_start"])
        end = int(section["page_end"])
        for unit in units:
            warnings = list(unit.review_warnings)
            if unit.unit_type not in allowed_types:
                warnings.append(f"unit_type 不在允许枚举中：{unit.unit_type}")
            refs = list(unit.source_refs)
            if not refs:
                warnings.append("缺少来源页码和原文摘录")
            for ref in refs:
                if ref.page < start or ref.page > end:
                    warnings.append(f"来源页码 {ref.page} 不在本章节范围内")
                    continue
                source_text = page_texts.get(ref.page, "")
                quote = ref.quote.strip()
                if not quote:
                    warnings.append(f"第 {ref.page} 页缺少原文摘录")
                elif (
                    source_text
                    and _normalise_for_match(quote)
                    not in _normalise_for_match(source_text)
                ):
                    warnings.append(
                        f"第 {ref.page} 页的原文摘录未能精确匹配 OCR/提取文本"
                    )
            checked.append(unit.model_copy(update={"review_warnings": warnings}))
        return checked

    def _request_units(
        self,
        *,
        material_type: str,
        section: dict[str, Any],
        content: str,
        page_texts: dict[int, str],
    ) -> AIResult:
        prompt = self._structure_prompt(
            material_type=material_type,
            section=section,
            content=content,
        )
        raw, finish_reason = self._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "你是严谨的中文课程材料结构化助手。"
                        "只依据输入内容工作，并输出合法 JSON。"
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        try:
            parsed = AIResponse.model_validate(_extract_json(raw))
        except (ValueError, json.JSONDecodeError, ValidationError):
            repaired = self._repair_json(raw)
            parsed = AIResponse.model_validate(_extract_json(repaired))
            raw = f"{raw}\n\n--- repaired ---\n{repaired}"
        units = self._validate_units(
            parsed.units,
            material_type=material_type,
            section=section,
            page_texts=page_texts,
        )
        return AIResult(
            units=units,
            raw_response=raw,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _split_content(content: str) -> tuple[str, str] | None:
        page_positions = [
            match.start() for match in re.finditer(r"(?=【第 \d+ 页】)", content)
        ]
        if len(page_positions) > 1:
            midpoint = len(content) // 2
            cut = min(page_positions[1:], key=lambda pos: abs(pos - midpoint))
            if 0 < cut < len(content):
                return content[:cut].strip(), content[cut:].strip()
        paragraphs = [part for part in re.split(r"\n{2,}", content) if part.strip()]
        if len(paragraphs) < 2:
            return None
        midpoint = len(content) // 2
        current_length = 0
        for index, paragraph in enumerate(paragraphs[:-1]):
            current_length += len(paragraph)
            if current_length >= midpoint:
                return (
                    "\n\n".join(paragraphs[: index + 1]),
                    "\n\n".join(paragraphs[index + 1 :]),
                )
        return None

    def structure_section(
        self,
        *,
        material_type: str,
        section: dict[str, Any],
        page_texts: dict[int, str],
    ) -> AIResult:
        def request_recursive(content: str, depth: int = 0) -> AIResult:
            if depth > 12:
                raise AIRequestError("章节分段层级过深，已停止以避免无限拆分")
            limit_error: AIRequestError | None = None
            try:
                result = self._request_units(
                    material_type=material_type,
                    section=section,
                    content=content,
                    page_texts=page_texts,
                )
            except AIRequestError as exc:
                message = str(exc).lower()
                if not any(
                    marker in message
                    for marker in (
                        "context",
                        "maximum context",
                        "too many tokens",
                        "token limit",
                        "request too large",
                        "输入过长",
                    )
                ):
                    raise
                result = None
                limit_error = exc

            if result is not None and result.finish_reason != "length":
                return result
            parts = self._split_content(content)
            if not parts:
                detail = limit_error or "AI 输出被截断"
                raise AIRequestError(f"{detail}，且该内容无法继续按段切分")
            first = request_recursive(parts[0], depth + 1)
            second = request_recursive(parts[1], depth + 1)
            return AIResult(
                units=[*first.units, *second.units],
                raw_response="\n\n--- segment ---\n\n".join(
                    [first.raw_response, second.raw_response]
                ),
                segmented=True,
            )

        return request_recursive(str(section["content"]))

    def infer_section_outline(
        self,
        *,
        material_type: str,
        pages: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        content = "\n\n".join(
            f"【第 {page['page_number']} 页】\n{page['final_text']}"
            for page in pages
            if str(page.get("final_text") or "").strip()
        )
        prompt = f"""
你负责为一份没有可靠标题层级的{material_type}材料恢复章节边界。
只依据输入的逐页文本，不要生成材料以外的标题。输出合法 JSON：
{{"sections":[{{"title":"标题","page_start":1,"page_end":2}}]}}
章节必须按页码顺序覆盖输入中的有效页面，不要把一道例题或一个连续要求拆散。
不输出 Markdown，不输出解释。

{content}
""".strip()
        raw, _ = self._complete(
            [
                {
                    "role": "system",
                    "content": "你只输出包含 sections 数组的合法 JSON。",
                },
                {"role": "user", "content": prompt},
            ]
        )
        try:
            data = _extract_json(raw)
        except (ValueError, json.JSONDecodeError):
            raw = self._repair_json(raw)
            data = _extract_json(raw)
        outline = data.get("sections")
        if not isinstance(outline, list) or not outline:
            raise AIRequestError("AI 未能识别出章节边界")
        valid_pages = {int(page["page_number"]) for page in pages}
        result: list[dict[str, Any]] = []
        for item in outline:
            start = int(item["page_start"])
            end = int(item["page_end"])
            title = str(item["title"]).strip()
            if not title or start > end:
                raise AIRequestError("AI 返回了无效的章节范围")
            if start not in valid_pages or end not in valid_pages:
                raise AIRequestError("AI 返回了输入中不存在的页码")
            result.append({"title": title, "page_start": start, "page_end": end})
        result.sort(key=lambda item: item["page_start"])
        return result
