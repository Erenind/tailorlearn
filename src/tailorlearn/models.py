from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MaterialType(StrEnum):
    TEXTBOOK = "textbook"
    STANDARD = "curriculum_standard"


class UnitStatus(StrEnum):
    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    VERIFIED = "verified"


class SourceKind(StrEnum):
    TEXT = "text"
    OCR = "ocr"
    EDITED = "edited"
    FAILED = "failed"


class PageResult(BaseModel):
    page_number: int
    raw_text: str
    final_text: str
    source_kind: str
    confidence: float | None = None
    status: str = "ok"
    warning_reason: str | None = None

    @property
    def char_count(self) -> int:
        return len(self.final_text)


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page: int
    quote: str = ""
    block_id: str | None = None


class AIUnit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    unit_type: str
    title: str
    content: str
    source_refs: list[SourceRef] = Field(default_factory=list)
    related_unit_ids: list[str] = Field(default_factory=list)
    review_warnings: list[str] = Field(default_factory=list)

    @field_validator("title", "content")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class AIResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    units: list[AIUnit]


class ProviderProfile(BaseModel):
    id: str
    vendor_name: str
    base_url: str
    model: str
    api_key: str = ""

    @property
    def masked_api_key(self) -> str:
        if not self.api_key:
            return ""
        key = self.api_key.strip()
        if len(key) <= 8:
            return "•" * max(len(key), 4)
        prefix = key[:3] if key.startswith("sk-") else ""
        return f"{prefix}••••••••{key[-4:]}"

    def public_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["api_key"] = self.masked_api_key
        return data


class ProviderConfig(BaseModel):
    active_profile_id: str | None = None
    profiles: list[ProviderProfile] = Field(default_factory=list)


class ExtractedSection(BaseModel):
    id: str
    document_id: str
    parent_id: str | None = None
    path: str
    title: str
    level: int
    order_index: int
    page_start: int
    page_end: int
    content: str
    split_method: str = "heading-rules"


class AIResult(BaseModel):
    units: list[AIUnit]
    raw_response: str
    finish_reason: str | None = None
    segmented: bool = False
