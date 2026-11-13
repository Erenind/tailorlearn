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
    assert at.selectbox(key="teaching_chat_dataset").value == dataset["id"]
    assert at.chat_input(key="teaching_chat_input").disabled is False
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
        item.key == "teaching_chat_dataset" for item in at.selectbox
    )

    question = at.selectbox(key="teaching_chat_common_question")
    assert at.menu_button(
        key="teaching_question_combo_actions"
    ).options == ["新增问题", "编辑当前问题", "删除当前问题"]
    student_question = next(
        option
        for option in question.options
        if option.endswith("具体分析学生D01等")
    )
    assert any("预设" in option for option in question.options)
    question.select(student_question).run()
    assert not at.exception
    assert at.chat_input(key="teaching_chat_input").value == "具体分析学生D01等"
    at.chat_input(key="teaching_chat_input").upload(
        "临时资料.txt",
        "这是一份临时教学素材".encode(),
    )
    st.cache_resource.clear()
