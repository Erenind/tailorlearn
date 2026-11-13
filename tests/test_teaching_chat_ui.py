from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.paths import AppPaths
from tailorlearn.providers import ProviderStore
from tailorlearn.teaching_chat import TeachingAssistantService
from tailorlearn.analytics import AnalyticsService
from tailorlearn.outline import OutlineService


def test_streamlit_teaching_chat_composer_and_pure_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    learning = LearningDataService(repository, paths.db_dir)
    dataset = learning.create_dataset("对话测试数据")
    learning.create_record(
        dataset["id"],
        "student",
        {"student_code": "D01", "class_code": "DC01", "grade": 7},
    )
    profile = ProviderStore(paths.provider_file).upsert(
        profile_id=None,
        vendor_name="Test",
        base_url="https://example.test/v1",
        model="test-model",
        api_key="test-key",
    )
    ProviderStore(paths.provider_file).set_active(profile.id)
    service = TeachingAssistantService(
        repository,
        learning,
        AnalyticsService(repository),
        OutlineService(paths, repository),
    )
    for index in range(1, 26):
        service.append_message(
            "user" if index % 2 else "assistant",
            f"历史消息 {index}",
        )

    monkeypatch.setattr(
        AppPaths,
        "default",
        classmethod(lambda cls: paths),
    )
    st.cache_resource.clear()
    app_file = Path(__file__).resolve().parents[1] / "src" / "tailorlearn" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=20).run()

    at.button(key="nav_teaching").click().run()
    assert not at.exception
    assert at.pills(key="teaching_chat_pure_mode").value is None
    assert at.button(
        key=f"teaching_dataset_choose_{dataset['id']}"
    ).label.endswith("当前")
    assert at.text_area(key="teaching_chat_draft").value == ""
    assert at.button(key="teaching_chat_send").disabled is False
    assert at.button(
        key="teaching_question_use_preset_student"
    ).label == "具体分析学生D01等"
    assert at.button(
        key="teaching_question_edit_preset_student"
    )
    assert at.button(
        key="teaching_question_delete_preset_student"
    )
    assert not any(
        item.value in {"班级主题证据", "AI 教学建议草稿"}
        for item in at.subheader
    )
    visible_markdown = "\n".join(item.value for item in at.markdown)
    assert "历史消息 6" in visible_markdown
    assert "历史消息 25" in visible_markdown
    assert "历史消息 5" not in visible_markdown

    at.button(key="teaching_load_earlier").click().run()
    assert not at.exception
    visible_markdown = "\n".join(item.value for item in at.markdown)
    assert "历史消息 1" in visible_markdown

    at.session_state["teaching_chat_pending_action"] = {
        "id": "pending_test",
        "tool": "write-data",
        "arguments": {
            "dataset_id": dataset["id"],
            "data_type": "student",
            "action": "create",
            "row_id": None,
            "fields": {
                "student_code": "D02",
                "class_code": "DC01",
                "grade": 7,
            },
        },
        "summary": "将新增 1 条模拟学生记录",
        "changes": {
            "student_code": {"before": None, "after": "D02"},
        },
    }
    at.run()
    assert at.button(key="teaching_pending_confirm")
    at.button(key="teaching_pending_confirm").click().run()
    assert not at.exception
    assert {
        item["student_code"]
        for item in learning.list_records(dataset["id"], "student")
    } == {"D01", "D02"}

    at.pills(key="teaching_chat_pure_mode").set_value("纯净模式").run()
    assert not at.exception
    assert not any(
        (item.key or "").startswith("teaching_dataset_choose_")
        for item in at.button
    )

    at.button(key="teaching_question_use_preset_student").click().run()
    assert not at.exception
    assert at.text_area(key="teaching_chat_draft").value == "具体分析学生D01等"
    assert at.file_uploader(key="teaching_temporary_upload")
    at.file_uploader(key="teaching_temporary_upload").upload(
        "临时资料.txt",
        "这是一份临时教学素材".encode(),
    ).run()
    assert not at.exception
    temporary_files = at.session_state["teaching_chat_temporary_files"]
    assert temporary_files[0]["name"] == "临时资料.txt"
    assert not repr(at).count(":violet-badge[预设]")
    st.cache_resource.clear()
