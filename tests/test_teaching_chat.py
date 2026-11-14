from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tailorlearn.analytics import AnalyticsService
from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService
from tailorlearn.models import ProviderProfile
from tailorlearn.outline import OutlineService
from tailorlearn.paths import AppPaths
from tailorlearn.teaching_chat import (
    AssistantCompletion,
    AssistantStreamCompletion,
    AssistantTextDelta,
    OpenAICompatibleAssistantClient,
    TeachingContentDelta,
    TeachingAssistantService,
    TeachingToolFinished,
    TeachingToolStarted,
    TeachingTurnComplete,
    TeachingToolContext,
    ToolCall,
)


class _ScriptedClient:
    def __init__(self, responses: list[AssistantCompletion]):
        self.responses = list(responses)
        self.message_snapshots: list[list[dict[str, object]]] = []

    def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> AssistantCompletion:
        _ = tools
        self.message_snapshots.append([dict(message) for message in messages])
        return self.responses.pop(0)


def _profile() -> ProviderProfile:
    return ProviderProfile(
        id="provider_test",
        vendor_name="Test",
        base_url="https://example.test/v1",
        model="test-model",
        api_key="test-key",
    )


def _service(
    tmp_path: Path,
) -> tuple[
    Repository,
    LearningDataService,
    TeachingAssistantService,
]:
    paths = AppPaths(tmp_path)
    paths.ensure()
    repository = Repository(paths.database_file)
    repository.initialize()
    learning = LearningDataService(repository, paths.db_dir)
    analytics = AnalyticsService(repository)
    outline = OutlineService(paths, repository)
    service = TeachingAssistantService(
        repository,
        learning,
        analytics,
        outline,
    )
    return repository, learning, service


def test_teaching_chat_suggestions_include_editable_presets(tmp_path: Path) -> None:
    repository, _learning, service = _service(tmp_path)

    suggestions = service.list_suggestions()
    assert len(suggestions) == 6
    assert suggestions[0]["text"] == "根据苏格拉底教学方式提教学方案"

    service.update_suggestion("preset_socratic", "用苏格拉底式提问设计教学")
    service.delete_suggestion("preset_practice")
    added = service.add_suggestion("根据本周错题设计讲评课")

    restarted = TeachingAssistantService(
        repository,
        _learning,
        AnalyticsService(repository),
        OutlineService(AppPaths(tmp_path), repository),
    )
    texts = {item["text"] for item in restarted.list_suggestions()}
    assert "用苏格拉底式提问设计教学" in texts
    assert "以实践为重点提个教学建议" not in texts
    assert "根据本周错题设计讲评课" in texts
    assert added["text"] == "根据本周错题设计讲评课"


def test_teaching_chat_database_migrates_tool_trace_column(
    tmp_path: Path,
) -> None:
    database_file = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_file) as connection:
        connection.execute(
            """
            CREATE TABLE teaching_chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    repository = Repository(database_file)
    repository.initialize()

    with repository.connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(teaching_chat_messages)"
            )
        }
    assert "tool_trace_json" in columns


def test_teaching_chat_keeps_history_pages_and_compresses_old_messages(
    tmp_path: Path,
) -> None:
    _repository, _learning, service = _service(tmp_path)
    for index in range(1, 15):
        service.append_message(
            "user" if index % 2 else "assistant",
            f"历史消息 {index}",
        )

    client = _ScriptedClient(
        [AssistantCompletion(content="用户偏好实践教学；待完成复习计划。")]
    )
    result = service.maybe_compress(_profile(), client)

    assert result is not None
    assert result["compressed_messages"] == 8
    assert len(service.list_messages(limit=20)) == 14
    with service.repo.connect() as connection:
        memory = connection.execute(
            "SELECT * FROM teaching_chat_memory WHERE id='default'"
        ).fetchone()
    assert "实践教学" in memory["summary"]
    assert memory["covered_message_id"] == 8


def test_teaching_chat_deletes_a_whole_turn_and_invalidates_memory(
    tmp_path: Path,
) -> None:
    _repository, _learning, service = _service(tmp_path)
    user_message_ids: list[int] = []
    for index in range(1, 8):
        user_message_ids.append(
            service.append_message("user", f"第 {index} 轮问题")["id"]
        )
        service.append_message("assistant", f"第 {index} 轮回复")

    client = _ScriptedClient(
        [AssistantCompletion(content="保留了前面的教学偏好。")]
    )
    assert service.maybe_compress(_profile(), client) is not None

    deleted = service.delete_conversation_turn(user_message_ids[1])

    assert deleted == 2
    remaining = service.list_messages(limit=20)
    assert [message["content"] for message in remaining] == [
        content
        for index in range(1, 8)
        if index != 2
        for content in (f"第 {index} 轮问题", f"第 {index} 轮回复")
    ]
    with service.repo.connect() as connection:
        memory = connection.execute(
            "SELECT * FROM teaching_chat_memory WHERE id='default'"
        ).fetchone()
    assert memory["summary"] == ""
    assert memory["covered_message_id"] == 0


def test_teaching_chat_tools_respect_pure_mode_and_write_confirmation(
    tmp_path: Path,
) -> None:
    _repository, learning, service = _service(tmp_path)
    dataset = learning.create_dataset("智能体数据")
    client = _ScriptedClient([])
    context = TeachingToolContext(
        dataset_id=dataset["id"],
        pure_mode=False,
        temporary_materials=[],
    )

    with pytest.raises(ValueError, match="纯净模式"):
        service.execute_tool(
            "read-data",
            {"data_type": "students"},
            TeachingToolContext(
                dataset_id=dataset["id"],
                pure_mode=True,
                temporary_materials=[],
            ),
            client=client,
            profile=_profile(),
        )
    with pytest.raises(ValueError, match="纯净模式"):
        service.execute_tool(
            "create-learning-list",
            {"mode": "theme", "theme_codes": ["T01"]},
            TeachingToolContext(
                dataset_id=dataset["id"],
                pure_mode=True,
                temporary_materials=[],
            ),
            client=client,
            profile=_profile(),
        )

    preview = service.execute_tool(
        "write-data",
        {
            "data_type": "student",
            "action": "create",
            "fields": {
                "student_code": "D01",
                "class_code": "DC01",
                "grade": 7,
            },
        },
        context,
        client=client,
        profile=_profile(),
    )
    assert preview["status"] == "pending_confirmation"
    assert learning.list_records(dataset["id"], "student") == []

    assert context.pending_action is not None
    written = service.execute_pending_action(
        context.pending_action,
        _profile(),
    )
    assert written["status"] == "written"
    students = learning.list_records(dataset["id"], "student")
    assert len(students) == 1
    assert students[0]["student_code"] == "D01"


def test_teaching_chat_agent_executes_read_tool_then_answers(
    tmp_path: Path,
) -> None:
    _repository, learning, service = _service(tmp_path)
    dataset = learning.create_dataset("智能体对话数据")
    learning.create_record(
        dataset["id"],
        "student",
        {"student_code": "D01", "class_code": "DC01", "grade": 7},
    )
    client = _ScriptedClient(
        [
            AssistantCompletion(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="read-data",
                        arguments={"data_type": "students"},
                    )
                ],
            ),
            AssistantCompletion(
                content="建议围绕 D01 所在班级组织分层练习。"
            ),
        ]
    )

    result = service.run_turn(
        prompt="给 DC01 一个教学建议",
        profile=_profile(),
        client=client,
        dataset_id=dataset["id"],
        pure_mode=False,
        temporary_materials=[],
    )

    assert "D01" in result.message["content"]
    messages = service.list_messages(limit=20)
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
    ]
    assert not any(
        "test-key" in json.dumps(snapshot, ensure_ascii=False)
        for snapshot in client.message_snapshots
    )
    system_prompt = str(client.message_snapshots[0][0]["content"])
    assert "行内公式写成 $...$" in system_prompt
    assert "Markdown 表格" in system_prompt


def test_teaching_chat_stream_emits_tool_events_and_content_deltas(
    tmp_path: Path,
) -> None:
    _repository, learning, service = _service(tmp_path)
    dataset = learning.create_dataset("流式对话数据")
    learning.create_record(
        dataset["id"],
        "student",
        {"student_code": "D01", "class_code": "DC01", "grade": 7},
    )
    client = _ScriptedClient(
        [
            AssistantCompletion(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_stream",
                        name="read-data",
                        arguments={"data_type": "students"},
                    )
                ],
            ),
            AssistantCompletion(
                content="结论：$x^2+1>0$。\n\n| 学生 | 班级 |\n|---|---|\n| D01 | DC01 |"
            ),
        ]
    )

    events = list(
        service.run_turn_stream(
            prompt="读取学生数据并用公式说明",
            profile=_profile(),
            client=client,
            dataset_id=dataset["id"],
            pure_mode=False,
            temporary_materials=[],
        )
    )

    started = next(
        event for event in events if isinstance(event, TeachingToolStarted)
    )
    finished = next(
        event for event in events if isinstance(event, TeachingToolFinished)
    )
    content = "".join(
        event.text
        for event in events
        if isinstance(event, TeachingContentDelta)
    )
    completed = next(
        event for event in events if isinstance(event, TeachingTurnComplete)
    )

    assert started.label == "读取学情数据"
    assert started.detail is not None
    assert "读取数据：学生" in started.detail
    assert "数据集：流式对话数据" in started.detail
    assert "最多返回：50 条" in started.detail
    assert finished.success is True
    assert finished.detail is not None
    assert "读取数据：学生" in finished.detail
    assert "数据集：流式对话数据" in finished.detail
    assert "记录数：1" in finished.detail
    assert "字段：学生编号、模拟班级编号、年级" in finished.detail
    assert "$x^2+1>0$" in content
    assert "| 学生 | 班级 |" in content
    assert content == completed.result.message["content"]
    assert completed.result.tool_trace[0]["label"] == "读取学情数据"
    assert completed.result.tool_trace[0]["progress"] == started.detail
    assert completed.result.tool_trace[0]["detail"] == finished.detail
    stored_assistant = service.list_messages(limit=10)[-1]
    assert stored_assistant["tool_trace"][0]["status"] == "complete"
    assert stored_assistant["tool_trace"][0]["detail"] == finished.detail


def test_ai_tool_progress_and_result_summaries_cover_all_tool_types(
    tmp_path: Path,
) -> None:
    _repository, learning, service = _service(tmp_path)
    dataset = learning.create_dataset("工具摘要数据")
    context = TeachingToolContext(
        dataset_id=dataset["id"],
        pure_mode=False,
        temporary_materials=[],
    )

    index_progress = service._tool_progress_detail(
        "get-book-index",
        {
            "query": "实数",
            "material_type": "textbook",
            "limit": 5,
        },
        context,
    )
    index_result = service._tool_trace_detail(
        "get-book-index",
        {"query": "实数"},
        {"matches": [{"title": "实数"}], "truncated": True},
    )
    book_progress = service._tool_progress_detail(
        "read-book",
        {"section_ids": ["S1", "S2"], "max_chars": 8000},
        context,
    )
    book_result = service._tool_trace_detail(
        "read-book",
        {},
        {
            "sections": [
                {
                    "title": "实数",
                    "page_start": 1,
                    "page_end": 2,
                    "content": "正文",
                    "truncated": False,
                }
            ]
        },
    )
    write_progress = service._tool_progress_detail(
        "write-data",
        {
            "data_type": "student",
            "action": "create",
            "dataset_id": dataset["id"],
            "fields": {"student_code": "D01"},
        },
        context,
    )
    write_result = service._tool_trace_detail(
        "write-data",
        {},
        {
            "status": "pending_confirmation",
            "preview": {"summary": "将新增 1 条学生记录"},
        },
    )
    compress_result = service._tool_trace_detail(
        "compress-record",
        {},
        {"compressed_messages": 8, "covered_message_id": 12},
    )

    assert index_progress is not None and "检索范围：课本" in index_progress
    assert index_result is not None
    assert "匹配章节：1 条" in index_result
    assert "结果已截断" in index_result
    assert book_progress is not None and "读取章节：2 个" in book_progress
    assert book_result is not None
    assert "实数（第 1-2 页）" in book_result
    assert write_progress is not None
    assert "操作：新增模拟学生" in write_progress
    assert "数据集：工具摘要数据" in write_progress
    assert "字段：学生编号" in write_progress
    assert write_result is not None
    assert "等待用户确认" in write_result
    assert compress_result == "压缩消息：8 条；覆盖至消息 12"


def test_openai_compatible_stream_accumulates_text_and_tool_calls() -> None:
    class _Completions:
        def create(self, **kwargs: object) -> list[SimpleNamespace]:
            assert kwargs["stream"] is True
            if kwargs.get("tools"):
                return [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id="call_1",
                                            function=SimpleNamespace(
                                                name="read-data",
                                                arguments='{"data_type":',
                                            ),
                                        )
                                    ],
                                )
                            )
                        ]
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[
                                        SimpleNamespace(
                                            index=0,
                                            id=None,
                                            function=SimpleNamespace(
                                                name=None,
                                                arguments='"students"}',
                                            ),
                                        )
                                    ],
                                )
                            )
                        ]
                    ),
                ]
            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content="结论：",
                                tool_calls=[],
                            )
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content="$x^2$。",
                                tool_calls=[],
                            )
                        )
                    ]
                ),
            ]

    client = OpenAICompatibleAssistantClient(_profile())
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())
    )

    text_events = list(client.stream([{"role": "user", "content": "问题"}]))
    assert "".join(
        event.text for event in text_events if isinstance(event, AssistantTextDelta)
    ) == "结论：$x^2$。"
    text_completion = text_events[-1]
    assert isinstance(text_completion, AssistantStreamCompletion)
    assert text_completion.completion.content == "结论：$x^2$。"

    tool_events = list(
        client.stream(
            [{"role": "user", "content": "读取数据"}],
            [{"type": "function"}],
        )
    )
    tool_completion = tool_events[-1]
    assert isinstance(tool_completion, AssistantStreamCompletion)
    assert tool_completion.completion.tool_calls[0].arguments == {
        "data_type": "students"
    }


def test_teaching_chat_read_data_includes_raw_answer_and_uses_memory(
    tmp_path: Path,
) -> None:
    _repository, learning, service = _service(tmp_path)
    dataset = learning.create_dataset("原始作答数据")
    learning.create_record(
        dataset["id"],
        "answer",
        {
            "record_code": "R001",
            "student_code": "D01",
            "material_code": "Q001",
            "theme_code": "T01",
            "node_code": "N01",
            "answer_event_code": "E001",
            "revision": 1,
            "submitted_at": "2026-01-20T09:00:00",
            "grading_status": "已评分",
            "score": 1,
            "full_score": 2,
            "final_answer": "学生完整的原始作答",
        },
    )
    context = TeachingToolContext(
        dataset_id=dataset["id"],
        pure_mode=False,
        temporary_materials=[],
    )
    result = service.execute_tool(
        "read-data",
        {"data_type": "answers"},
        context,
        client=_ScriptedClient([]),
        profile=_profile(),
    )
    assert result["rows"][0]["final_answer"] == "学生完整的原始作答"

    service._save_memory("用户偏好先实践后总结。", 0)
    client = _ScriptedClient([AssistantCompletion(content="收到。")])
    service.run_turn(
        prompt="继续按我的偏好设计",
        profile=_profile(),
        client=client,
        dataset_id=dataset["id"],
        pure_mode=False,
        temporary_materials=[],
    )
    assert any(
        "长期压缩记忆" in str(message.get("content"))
        for message in client.message_snapshots[0]
    )


def test_create_learning_list_previews_then_delegates_to_outline_agent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _repository, learning, service = _service(tmp_path)
    dataset = learning.create_dataset("主题模式数据")
    standard = {
        "id": "set_standard",
        "name": "课程标准",
        "material_type": "curriculum_standard",
    }
    textbook = {
        "id": "set_textbook",
        "name": "数学课本",
        "material_type": "textbook",
    }

    def materials(material_type: str) -> list[dict[str, object]]:
        if material_type == "curriculum_standard":
            return [standard]
        return [textbook]

    def documents(material_id: str) -> list[dict[str, object]]:
        return [
            {
                "id": (
                    "doc_standard"
                    if material_id == "set_standard"
                    else "doc_textbook"
                )
            }
        ]

    captured: dict[str, object] = {}

    def prepare_two_stage_scope(**kwargs: object) -> dict[str, object]:
        captured["scope_args"] = kwargs
        return {"mode": kwargs["mode"]}

    def select_materials(
        *,
        scope: dict[str, object],
        profile: ProviderProfile,
    ) -> dict[str, object]:
        _ = scope, profile
        return {
            "standard_section_ids": ["std_section"],
            "textbook_section_ids": ["book_section"],
            "reasons": ["匹配主题"],
        }

    def generate_from_selection(**kwargs: object) -> dict[str, object]:
        captured["generate_args"] = kwargs
        return {
            "run": {
                "id": "run_created",
                "name": "主题模式学习清单",
            },
            "items": [object(), object()],
            "relations": [object()],
        }

    monkeypatch.setattr(service.outline, "materials", materials)
    monkeypatch.setattr(service.outline, "documents", documents)
    monkeypatch.setattr(
        service.outline,
        "prepare_two_stage_scope",
        prepare_two_stage_scope,
    )
    monkeypatch.setattr(service.outline, "select_materials", select_materials)
    monkeypatch.setattr(
        service.outline,
        "generate_from_selection",
        generate_from_selection,
    )
    context = TeachingToolContext(
        dataset_id=dataset["id"],
        pure_mode=False,
        temporary_materials=[],
    )

    preview = service.execute_tool(
        "create-learning-list",
        {"mode": "theme", "theme_codes": ["T01"]},
        context,
        client=_ScriptedClient([]),
        profile=_profile(),
    )
    assert preview["status"] == "pending_confirmation"
    assert context.pending_action is not None
    result = service.execute_pending_action(context.pending_action, _profile())

    assert result == {
        "status": "created",
        "run_id": "run_created",
        "run_name": "主题模式学习清单",
        "item_count": 2,
        "relation_count": 1,
    }
    scope_args = captured["scope_args"]
    assert isinstance(scope_args, dict)
    assert scope_args["dataset_id"] == dataset["id"]
    assert scope_args["theme_codes"] == ["T01"]
