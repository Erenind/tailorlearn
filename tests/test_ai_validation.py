from __future__ import annotations

from tailorlearn.ai import OpenAICompatibleClient, _extract_json
from tailorlearn.models import AIUnit, SourceRef
from tailorlearn.models import ProviderProfile


def test_extract_json_accepts_fenced_response():
    assert _extract_json('```json\n{"units": []}\n```') == {"units": []}


def test_validation_marks_cross_page_and_unmatched_quotes_for_review():
    client = object.__new__(OpenAICompatibleClient)
    section = {"page_start": 1, "page_end": 2}
    unit = AIUnit(
        unit_type="concept",
        title="测试",
        content="内容",
        source_refs=[
            SourceRef(page=1, quote="原文"),
            SourceRef(page=3, quote="越界"),
            SourceRef(page=2, quote="不存在"),
        ],
    )
    checked = client._validate_units(
        [unit],
        material_type="textbook",
        section=section,
        page_texts={1: "这是原文", 2: "另一段内容"},
    )[0]
    assert any("不在本章节范围" in warning for warning in checked.review_warnings)
    assert any("未能精确匹配" in warning for warning in checked.review_warnings)


def test_validation_rejects_material_incompatible_unit_type():
    client = object.__new__(OpenAICompatibleClient)
    unit = AIUnit(
        unit_type="content_requirement",
        title="测试",
        content="内容",
        source_refs=[SourceRef(page=1, quote="原文")],
    )
    checked = client._validate_units(
        [unit],
        material_type="textbook",
        section={"page_start": 1, "page_end": 1},
        page_texts={1: "原文"},
    )[0]
    assert any("不在允许枚举" in warning for warning in checked.review_warnings)


def test_student_diagnosis_client_parses_a_structured_draft(monkeypatch):
    def fake_complete(self, messages):
        return (
            """{
            "summary":"有效记录不足，建议补充观察。",
            "observations":[],
            "information_gaps":[{"theme_code":"T01","statement":"仅一条有效记录。"}],
            "suggestions":[{"material_code":"Q002","reason":"补充同一指标。"}],
            "teacher_actions":["补充观察后再更新。"]
            }""",
            "stop",
        )

    monkeypatch.setattr(OpenAICompatibleClient, "_complete", fake_complete)
    client = OpenAICompatibleClient(
        ProviderProfile(
            id="provider_test",
            vendor_name="Test",
            base_url="https://example.test/v1",
            model="test-model",
            api_key="test-key",
        )
    )

    draft, raw = client.draft_student_diagnosis(
        {
            "student_code": "D01",
            "node_code": "N01",
            "themes": [],
            "indicators": [],
            "evidence": {},
            "candidate_material_codes": ["Q002"],
        }
    )

    assert draft["summary"] == "有效记录不足，建议补充观察。"
    assert draft["suggestions"][0]["material_code"] == "Q002"
    assert raw.startswith("{")
