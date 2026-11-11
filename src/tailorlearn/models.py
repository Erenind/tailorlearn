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


class OutlineSourceKind(StrEnum):
    STANDARD = "standard"
    TEXTBOOK = "textbook"
    REQUIREMENT = "requirement"


class OutlineItemStatus(StrEnum):
    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    VERIFIED = "verified"


class OutlineRelationType(StrEnum):
    PREREQUISITE = "prerequisite"
    APPLICATION = "application"
    PARALLEL = "parallel"
    CONTRAST = "contrast"
    OTHER = "other"


class OutlineEvidence(BaseModel):
    """One verifiable citation: a page, a verbatim quote and its owner."""

    model_config = ConfigDict(extra="ignore")

    kind: str
    material_set_id: str = ""
    material_name: str = ""
    document_id: str = ""
    file_name: str = ""
    section_id: str = ""
    section_path: str = ""
    page: int | None = None
    page_end: int | None = None
    printed_page: str | None = None
    quote: str = ""
    unit_id: str | None = None
    verified: bool = False
    verification_note: str = ""


class OutlineRequirement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    clause: str = ""
    verb: str = ""
    level: str = ""
    text: str
    evidence: OutlineEvidence | None = None
    verified: bool = False
    note: str = ""

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class OutlineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    run_id: str = ""
    item_key: str
    ordinal: int = 0
    theme_code: str = ""
    theme_name: str = ""
    item_type: str = "knowledge_point"
    content: str
    key_points: list[str] = Field(default_factory=list)
    indicator_codes: list[str] = Field(default_factory=list)
    requirements: list[OutlineRequirement] = Field(default_factory=list)
    sources: list[OutlineEvidence] = Field(default_factory=list)
    notes: str = ""
    review_warnings: list[str] = Field(default_factory=list)
    status: str = OutlineItemStatus.DRAFT.value
    current_version: int = 1

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class OutlineRelation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    run_id: str = ""
    source_item_id: str = ""
    source_item_key: str = ""
    target_item_id: str = ""
    target_item_key: str = ""
    relation_type: str = OutlineRelationType.PREREQUISITE.value
    note: str = ""
    basis: str = ""
    evidence: list[OutlineEvidence] = Field(default_factory=list)
    origin: str = "rule"
    status: str = OutlineItemStatus.DRAFT.value
    created_at: str = ""


class OutlineDraftRequirement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    clause: str = ""
    level: str = ""
    standard_page: int | None = None
    quote: str = ""


class OutlineDraftSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: str = OutlineSourceKind.TEXTBOOK.value
    material_set_id: str = ""
    document_id: str = ""
    section_id: str = ""
    unit_id: str | None = None
    page: int | None = None
    quote: str = ""


class OutlineDraftItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item_key: str = ""
    theme_code: str = ""
    item_type: str = "knowledge_point"
    content: str
    key_points: list[str] = Field(default_factory=list)
    indicator_codes: list[str] = Field(default_factory=list)
    requirements: list[OutlineDraftRequirement] = Field(default_factory=list)
    sources: list[OutlineDraftSource] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class OutlineDraftRelation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_key: str
    target_key: str
    relation_type: str = OutlineRelationType.PREREQUISITE.value
    note: str = ""
    basis: str = ""


class OutlineDraftResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scope_note: str = ""
    items: list[OutlineDraftItem]
    relations: list[OutlineDraftRelation] = Field(default_factory=list)


class OutlineSelectionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    standard_section_ids: list[str] = Field(default_factory=list)
    textbook_section_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class DiagnosisObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    theme_code: str
    statement: str
    evidence_record_codes: list[str] = Field(default_factory=list)
    indicator_codes: list[str] = Field(default_factory=list)


class DiagnosisInformationGap(BaseModel):
    model_config = ConfigDict(extra="ignore")

    theme_code: str
    statement: str


class DiagnosisSuggestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    material_code: str
    reason: str


class StudentDiagnosisDraft(BaseModel):
    """Structured, teacher-reviewable response for one student's evidence pack."""

    model_config = ConfigDict(extra="ignore")

    summary: str
    observations: list[DiagnosisObservation] = Field(default_factory=list)
    information_gaps: list[DiagnosisInformationGap] = Field(default_factory=list)
    suggestions: list[DiagnosisSuggestion] = Field(default_factory=list)
    teacher_actions: list[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def summary_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("summary must not be blank")
        return value


class TeachingObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    theme_code: str
    statement: str
    indicator_codes: list[str] = Field(default_factory=list)


class TeachingInformationGap(BaseModel):
    model_config = ConfigDict(extra="ignore")

    theme_code: str
    indicator_code: str
    statement: str


class TeachingGroupActivity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    theme_code: str
    group: str
    activity: str
    student_codes: list[str] = Field(default_factory=list)


class TeachingMaterialSuggestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    theme_code: str
    indicator_code: str
    material_code: str
    student_codes: list[str] = Field(default_factory=list)
    reason: str


class TeachingRecommendationDraft(BaseModel):
    """Structured, teacher-reviewable response for one class evidence pack."""

    model_config = ConfigDict(extra="ignore")

    summary: str
    observations: list[TeachingObservation] = Field(default_factory=list)
    information_gaps: list[TeachingInformationGap] = Field(default_factory=list)
    group_activities: list[TeachingGroupActivity] = Field(default_factory=list)
    material_suggestions: list[TeachingMaterialSuggestion] = Field(default_factory=list)
    teacher_actions: list[str] = Field(default_factory=list)
    post_lesson_checks: list[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def summary_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("summary must not be blank")
        return value
