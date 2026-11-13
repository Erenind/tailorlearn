from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Protocol
import uuid

from openai import OpenAI

from tailorlearn.ai import AIRequestError
from tailorlearn.analytics import AnalyticsService
from tailorlearn.db import Repository, utc_now
from tailorlearn.learning_data import DATA_TYPE_SPECS, LearningDataService
from tailorlearn.models import ProviderProfile
from tailorlearn.outline import OutlineService


TEACHING_CHAT_API_VERSION = "2026-11-13-teaching-chat-agent-v1"
TEACHING_SUGGESTION_PRESETS = (
    ("preset_socratic", "根据苏格拉底教学方式提教学方案"),
    ("preset_practice", "以实践为重点提个教学建议"),
    ("preset_student", "具体分析学生D01等"),
    ("preset_class", "给班级DC01提建议"),
    ("preset_ebbinghaus", "根据艾宾浩斯遗忘曲线分析今天需要复习哪些内容"),
    ("preset_operations", "根据实数的运算这一章提供教学建议"),
)
MEMORY_MESSAGE_LIMIT = 12
MEMORY_CHAR_LIMIT = 24_000
MEMORY_KEEP_RECENT = 6
MAX_TOOL_ROUNDS = 8


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AssistantCompletion:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class TeachingAssistantClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AssistantCompletion: ...


class OpenAICompatibleAssistantClient:
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

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AssistantCompletion:
        kwargs: dict[str, Any] = {
            "model": self.profile.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise AIRequestError(str(exc)) from exc
        if not response.choices:
            raise AIRequestError("AI 服务没有返回任何结果")
        message = response.choices[0].message
        calls: list[ToolCall] = []
        for item in message.tool_calls or []:
            try:
                arguments = json.loads(item.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise AIRequestError("AI 工具参数不是合法 JSON") from exc
            if not isinstance(arguments, dict):
                raise AIRequestError("AI 工具参数必须是 JSON 对象")
            calls.append(
                ToolCall(
                    id=item.id,
                    name=item.function.name,
                    arguments=arguments,
                )
            )
        return AssistantCompletion(
            content=message.content,
            tool_calls=calls,
        )


@dataclass
class TeachingToolContext:
    dataset_id: str | None
    pure_mode: bool
    temporary_materials: list[dict[str, Any]]
    pending_action: dict[str, Any] | None = None


@dataclass(frozen=True)
class TeachingTurnResult:
    message: dict[str, Any]
    pending_action: dict[str, Any] | None
    compression: dict[str, Any] | None


class TeachingAssistantService:
    def __init__(
        self,
        repository: Repository,
        learning: LearningDataService,
        analytics: AnalyticsService,
        outline: OutlineService,
    ):
        self.repo = repository
        self.learning = learning
        self.analytics = analytics
        self.outline = outline
        self._ensure_suggestion_presets()

    # ------------------------------------------------------------------
    # conversation and memory persistence
    # ------------------------------------------------------------------
    def append_message(self, role: str, content: str) -> dict[str, Any]:
        content = content.strip()
        if role not in {"user", "assistant"}:
            raise ValueError("未知消息角色")
        if not content:
            raise ValueError("消息内容不能为空")
        created_at = utc_now()
        with self.repo.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO teaching_chat_messages(role, content, created_at)
                VALUES (?, ?, ?)
                """,
                (role, content, created_at),
            )
            message_id = int(cursor.lastrowid)
        return {
            "id": message_id,
            "role": role,
            "content": content,
            "created_at": created_at,
        }

    def list_messages(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self.repo.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT id, role, content, created_at
                    FROM teaching_chat_messages
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_messages(self) -> int:
        with self.repo.connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM teaching_chat_messages"
                ).fetchone()[0]
            )

    def _memory(self) -> dict[str, Any]:
        with self.repo.connect() as connection:
            row = connection.execute(
                "SELECT * FROM teaching_chat_memory WHERE id='default'"
            ).fetchone()
        if row is None:
            return {
                "summary": "",
                "covered_message_id": 0,
                "updated_at": "",
            }
        return dict(row)

    def _save_memory(self, summary: str, covered_message_id: int) -> None:
        with self.repo.connect() as connection:
            connection.execute(
                """
                INSERT INTO teaching_chat_memory
                (id, summary, covered_message_id, updated_at)
                VALUES ('default', ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    summary=excluded.summary,
                    covered_message_id=excluded.covered_message_id,
                    updated_at=excluded.updated_at
                """,
                (summary.strip(), int(covered_message_id), utc_now()),
            )

    def _messages_after_memory(self) -> list[dict[str, Any]]:
        memory = self._memory()
        with self.repo.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, created_at
                FROM teaching_chat_messages
                WHERE id>?
                ORDER BY id
                """,
                (memory["covered_message_id"],),
            ).fetchall()
        return [dict(row) for row in rows]

    def _compress_messages(
        self,
        profile: ProviderProfile,
        client: TeachingAssistantClient,
        *,
        prior_summary: str,
        messages: list[dict[str, Any]],
    ) -> str:
        transcript = "\n\n".join(
            f"[{message['role']}] {str(message['content'])[:3000]}"
            for message in messages
        )
        completion = client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "你负责压缩教师与教学建议智能体的对话记忆。"
                        "输出一份简洁、可继续使用的中文累计记忆，重点保留："
                        "用户的教学偏好、长期约束、已确认的数据范围、"
                        "已经完成的操作、未完成事项和关键决定。"
                        "删除寒暄、重复内容和完整学生敏感明细；"
                        "不要编造事实；不要输出 JSON；控制在 6000 字以内。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"已有压缩记忆：\n{prior_summary or '尚无'}\n\n"
                        f"需要合并的新对话：\n{transcript}"
                    ),
                },
            ]
        )
        summary = (completion.content or "").strip()
        if not summary:
            raise AIRequestError("记忆压缩没有返回内容")
        return summary[:6000]

    def maybe_compress(
        self,
        profile: ProviderProfile,
        client: TeachingAssistantClient,
        *,
        force: bool = False,
    ) -> dict[str, Any] | None:
        messages = self._messages_after_memory()
        character_count = sum(len(str(item["content"])) for item in messages)
        if not force and (
            len(messages) <= MEMORY_MESSAGE_LIMIT
            and character_count <= MEMORY_CHAR_LIMIT
        ):
            return None
        if len(messages) <= MEMORY_KEEP_RECENT:
            return None
        to_compress = messages[:-MEMORY_KEEP_RECENT]
        memory = self._memory()
        summary = self._compress_messages(
            profile,
            client,
            prior_summary=str(memory["summary"]),
            messages=to_compress,
        )
        covered_message_id = int(to_compress[-1]["id"])
        self._save_memory(summary, covered_message_id)
        return {
            "compressed_messages": len(to_compress),
            "covered_message_id": covered_message_id,
        }

    def _model_history(self) -> list[dict[str, Any]]:
        memory = self._memory()
        messages = self._messages_after_memory()
        result: list[dict[str, Any]] = []
        if memory["summary"]:
            result.append(
                {
                    "role": "system",
                    "content": f"长期压缩记忆：\n{memory['summary']}",
                }
            )
        result.extend(
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages[-MEMORY_MESSAGE_LIMIT:]
        )
        return result

    # ------------------------------------------------------------------
    # suggestions
    # ------------------------------------------------------------------
    def _ensure_suggestion_presets(self) -> None:
        now = utc_now()
        with self.repo.connect() as connection:
            for index, (suggestion_id, text) in enumerate(
                TEACHING_SUGGESTION_PRESETS,
                start=1,
            ):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO teaching_chat_suggestions
                    (id, text, is_preset, sort_order, created_at, updated_at)
                    VALUES (?, ?, 1, ?, ?, ?)
                    """,
                    (suggestion_id, text, index, now, now),
                )

    def list_suggestions(self) -> list[dict[str, Any]]:
        with self.repo.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM teaching_chat_suggestions
                WHERE deleted_at IS NULL
                ORDER BY sort_order, created_at, id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def add_suggestion(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("常用问题不能为空")
        suggestion_id = f"question_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        with self.repo.connect() as connection:
            sort_order = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sort_order), 0) + 1 "
                    "FROM teaching_chat_suggestions"
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO teaching_chat_suggestions
                (id, text, is_preset, sort_order, created_at, updated_at)
                VALUES (?, ?, 0, ?, ?, ?)
                """,
                (suggestion_id, text, sort_order, now, now),
            )
        return {"id": suggestion_id, "text": text}

    def update_suggestion(self, suggestion_id: str, text: str) -> None:
        text = text.strip()
        if not text:
            raise ValueError("常用问题不能为空")
        with self.repo.connect() as connection:
            updated = connection.execute(
                """
                UPDATE teaching_chat_suggestions
                SET text=?, updated_at=?
                WHERE id=? AND deleted_at IS NULL
                """,
                (text, utc_now(), suggestion_id),
            ).rowcount
        if not updated:
            raise ValueError("找不到该常用问题")

    def delete_suggestion(self, suggestion_id: str) -> None:
        with self.repo.connect() as connection:
            updated = connection.execute(
                """
                UPDATE teaching_chat_suggestions
                SET deleted_at=?, updated_at=?
                WHERE id=? AND deleted_at IS NULL
                """,
                (utc_now(), utc_now(), suggestion_id),
            ).rowcount
        if not updated:
            raise ValueError("找不到该常用问题")

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------
    @staticmethod
    def tool_definitions() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get-book-index",
                    "description": (
                        "搜索现有课本、课程标准及其目录。返回材料、文档和章节索引。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "material_type": {
                                "type": "string",
                                "enum": ["textbook", "curriculum_standard"],
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 200,
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read-book",
                    "description": "读取指定课本或课程标准章节的正文。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "section_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 12,
                            },
                            "max_chars": {
                                "type": "integer",
                                "minimum": 1000,
                                "maximum": 50000,
                            },
                        },
                        "required": ["section_ids"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read-data",
                    "description": (
                        "读取学情数据或已有学习内容梳理清单。"
                        "纯净模式不能读取学情数据，但可以读取学习内容梳理清单。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_type": {
                                "type": "string",
                                "enum": [
                                    "students",
                                    "classes",
                                    "themes",
                                    "requirements",
                                    "materials",
                                    "nodes",
                                    "answers",
                                    "quality",
                                    "outline_runs",
                                    "outline_items",
                                ],
                            },
                            "dataset_id": {"type": "string"},
                            "run_id": {"type": "string"},
                            "query": {"type": "string"},
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 200,
                            },
                        },
                        "required": ["data_type"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create-learning-list",
                    "description": (
                        "调用学习内容梳理子智能体创建新的学习内容清单。"
                        "纯净模式只能使用课本模式。执行前需要用户确认。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["textbook", "theme"],
                            },
                            "textbook_material_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "textbook_document_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "standard_material_id": {"type": "string"},
                            "standard_document_id": {"type": "string"},
                            "theme_codes": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "specific_requirements": {"type": "string"},
                            "name": {"type": "string"},
                        },
                        "required": ["mode"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write-data",
                    "description": (
                        "新增或修改具体学情数据。禁止删除。"
                        "执行前需要用户确认。纯净模式不可用。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_type": {
                                "type": "string",
                                "enum": [
                                    "student",
                                    "requirement",
                                    "material",
                                    "node",
                                    "answer",
                                ],
                            },
                            "action": {
                                "type": "string",
                                "enum": ["create", "update"],
                            },
                            "dataset_id": {"type": "string"},
                            "row_id": {"type": "integer"},
                            "fields": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["data_type", "action", "fields"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compress-record",
                    "description": (
                        "立即压缩较早的对话记忆。通常由系统自动调用。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _material_catalog(self) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for material_type in ("textbook", "curriculum_standard"):
            for material in self.repo.list_material_sets(
                material_type=material_type
            ):
                documents = []
                for document in self.repo.list_documents(material["id"]):
                    sections = [
                        {
                            "section_id": section["id"],
                            "title": section["title"],
                            "path": section["path"],
                            "page_start": section["page_start"],
                            "page_end": section["page_end"],
                        }
                        for section in self.repo.list_sections(document["id"])
                    ]
                    documents.append(
                        {
                            "document_id": document["id"],
                            "file_name": document["file_name"],
                            "page_count": document["page_count"],
                            "sections": sections,
                        }
                    )
                catalog.append(
                    {
                        "material_set_id": material["id"],
                        "name": material["name"],
                        "material_type": material_type,
                        "version": material["version"],
                        "documents": documents,
                    }
                )
        return catalog

    def _read_book(self, arguments: dict[str, Any]) -> dict[str, Any]:
        section_ids = [
            str(value)
            for value in arguments.get("section_ids", [])
            if str(value).strip()
        ][:12]
        if not section_ids:
            raise ValueError("read-book 需要至少一个 section_id")
        max_chars = max(
            1000,
            min(int(arguments.get("max_chars") or 16000), 50000),
        )
        remaining = max_chars
        result: list[dict[str, Any]] = []
        for section_id in section_ids:
            section = self.repo.get_section(section_id)
            if section is None:
                continue
            content = str(section["content"])
            returned = content[:remaining]
            result.append(
                {
                    "section_id": section_id,
                    "title": section["title"],
                    "path": section["path"],
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "content": returned,
                    "truncated": len(returned) < len(content),
                }
            )
            remaining -= len(returned)
            if remaining <= 0:
                break
        return {"sections": result}

    def _book_index(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip().casefold()
        material_type = arguments.get("material_type")
        limit = max(1, min(int(arguments.get("limit") or 100), 200))
        matches: list[dict[str, Any]] = []
        for material in self.repo.list_material_sets(
            material_type=str(material_type) if material_type else None
        ):
            material_matches = not query or query in material["name"].casefold()
            for document in self.repo.list_documents(material["id"]):
                document_matches = (
                    material_matches
                    or query in document["file_name"].casefold()
                )
                for section in self.repo.list_sections(document["id"]):
                    haystack = (
                        f"{material['name']} {document['file_name']} "
                        f"{section['title']} {section['path']}"
                    ).casefold()
                    if document_matches or not query or query in haystack:
                        matches.append(
                            {
                                "material_set_id": material["id"],
                                "material_name": material["name"],
                                "material_type": material["material_type"],
                                "document_id": document["id"],
                                "file_name": document["file_name"],
                                "section_id": section["id"],
                                "title": section["title"],
                                "path": section["path"],
                                "page_start": section["page_start"],
                                "page_end": section["page_end"],
                            }
                        )
                    if len(matches) >= limit:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def _read_data(self, arguments: dict[str, Any]) -> dict[str, Any]:
        data_type = str(arguments["data_type"])
        limit = max(1, min(int(arguments.get("limit") or 50), 200))
        query = str(arguments.get("query") or "").strip().casefold()
        if data_type in {"outline_runs", "outline_items"}:
            if data_type == "outline_runs":
                rows = self.outline.list_runs(
                    str(arguments["dataset_id"])
                    if arguments.get("dataset_id")
                    else None
                )[:limit]
                return {"data_type": data_type, "rows": rows}
            run_id = str(arguments.get("run_id") or "")
            if not run_id:
                raise ValueError("读取学习内容条目前需要 run_id")
            run = self.outline.get_run(run_id)
            if run is None:
                raise ValueError("找不到该学习内容梳理清单")
            items = [
                item.model_dump()
                for item in self.outline.items(run_id)
                if not query
                or query in json.dumps(item.model_dump(), ensure_ascii=False).casefold()
            ][:limit]
            return {
                "data_type": data_type,
                "run": run,
                "items": items,
            }
        if arguments.get("_pure_mode"):
            raise ValueError("纯净模式不能读取学情数据")
        dataset_id = str(
            arguments.get("dataset_id") or arguments.get("_active_dataset_id") or ""
        )
        if not dataset_id:
            raise ValueError("读取学情数据前需要选择学情数据集")
        if data_type == "classes":
            classes = sorted(
                {
                    str(item.get("class_code") or "未分班")
                    for item in self.learning.list_records(dataset_id, "student")
                }
            )
            rows: list[dict[str, Any]] = [
                {"class_code": class_code} for class_code in classes
            ]
        elif data_type == "quality":
            return self.learning.quality_report(dataset_id)
        else:
            singular = {
                "students": "student",
                "themes": "requirement",
                "requirements": "requirement",
                "materials": "material",
                "nodes": "node",
                "answers": "answer",
            }.get(data_type)
            if singular is None:
                raise ValueError("不支持的学情数据类型")
            rows = self.learning.list_records(
                dataset_id,
                singular,
                latest_answers=singular == "answer",
            )
        if query:
            rows = [
                row
                for row in rows
                if query in json.dumps(row, ensure_ascii=False).casefold()
            ]
        return {
            "dataset_id": dataset_id,
            "data_type": data_type,
            "rows": rows[:limit],
            "truncated": len(rows) > limit,
        }

    def _preview_create_learning_list(
        self,
        arguments: dict[str, Any],
        context: TeachingToolContext,
    ) -> dict[str, Any]:
        mode = str(arguments.get("mode") or "textbook")
        if mode not in {"textbook", "theme"}:
            raise ValueError("学习内容梳理模式无效")
        if context.pure_mode and mode != "textbook":
            raise ValueError("纯净模式只能创建课本模式学习内容清单")
        textbooks = self.outline.materials("textbook")
        selected_ids = [
            str(value)
            for value in arguments.get("textbook_material_ids", [])
            if str(value).strip()
        ] or [str(item["id"]) for item in textbooks]
        if not selected_ids:
            raise ValueError("尚无课本材料，无法创建学习内容清单")
        selected_names = [
            str(item["name"]) for item in textbooks if str(item["id"]) in selected_ids
        ]
        theme_codes = [
            str(value)
            for value in arguments.get("theme_codes", [])
            if str(value).strip()
        ]
        if mode == "theme" and not theme_codes:
            raise ValueError("主题模式需要 theme_codes")
        standard_materials = self.outline.materials("curriculum_standard")
        standard_id = str(
            arguments.get("standard_material_id")
            or (
                standard_materials[0]["id"]
                if standard_materials
                else ""
            )
        )
        standard = next(
            (
                item
                for item in standard_materials
                if str(item["id"]) == standard_id
            ),
            None,
        )
        if standard is None:
            raise ValueError("找不到所选课程标准材料")
        standard_documents = self.outline.documents(str(standard["id"]))
        standard_document_id = str(
            arguments.get("standard_document_id")
            or (standard_documents[0]["id"] if standard_documents else "")
        )
        if not any(
            str(item["id"]) == standard_document_id
            for item in standard_documents
        ):
            raise ValueError("找不到所选课程标准文档")
        return {
            "tool": "create-learning-list",
            "arguments": {
                **arguments,
                "mode": mode,
                "dataset_id": context.dataset_id,
                "textbook_material_ids": selected_ids,
                "textbook_document_ids": [
                    str(value)
                    for value in arguments.get("textbook_document_ids", [])
                    if str(value).strip()
                ],
                "standard_material_id": str(standard["id"]),
                "standard_document_id": standard_document_id,
                "theme_codes": theme_codes,
            },
            "summary": (
                f"将创建{mode == 'theme' and '主题' or '课本'}模式学习内容清单；"
                f"课程标准：{standard['name']}；"
                f"课本：{'、'.join(selected_names) or '未指定'}；"
                + (
                    f"主题：{'、'.join(theme_codes)}"
                    if theme_codes
                    else f"数据集：{context.dataset_id or '未选择'}"
                )
            ),
        }

    def _preview_write_data(
        self,
        arguments: dict[str, Any],
        context: TeachingToolContext,
    ) -> dict[str, Any]:
        if context.pure_mode:
            raise ValueError("纯净模式不能修改学情数据")
        data_type = str(arguments.get("data_type") or "")
        if data_type not in DATA_TYPE_SPECS:
            raise ValueError("不支持的学情数据类型")
        action = str(arguments.get("action") or "")
        if action not in {"create", "update"}:
            raise ValueError("write-data 只允许 create 或 update")
        dataset_id = str(arguments.get("dataset_id") or context.dataset_id or "")
        if not dataset_id:
            raise ValueError("修改学情数据前需要选择数据集")
        values = arguments.get("fields")
        if not isinstance(values, dict) or not values:
            raise ValueError("write-data 需要 fields")
        spec = DATA_TYPE_SPECS[data_type]
        allowed_fields = {field.key for field in spec.fields}
        unknown = sorted(set(values) - allowed_fields)
        if unknown:
            raise ValueError(f"包含不支持的字段：{'、'.join(unknown)}")
        preview_fields = {
            key: value for key, value in values.items() if key in allowed_fields
        }
        row_id = arguments.get("row_id")
        if action == "update":
            if row_id is None:
                raise ValueError("更新学情数据需要 row_id")
            current = self.learning.get_record(dataset_id, data_type, int(row_id))
            if current is None:
                raise ValueError("找不到要修改的学情记录")
            changes = {
                key: {
                    "before": current.get(key),
                    "after": value,
                }
                for key, value in preview_fields.items()
                if current.get(key) != value
            }
            if not changes:
                raise ValueError("没有检测到需要修改的字段")
            summary = (
                f"将修改 {spec.label} row_id={row_id}，"
                f"共 {len(changes)} 个字段"
            )
        else:
            changes = {
                key: {"before": None, "after": value}
                for key, value in preview_fields.items()
            }
            summary = f"将新增 1 条 {spec.label} 记录"
        return {
            "tool": "write-data",
            "arguments": {
                **arguments,
                "dataset_id": dataset_id,
                "row_id": int(row_id) if row_id is not None else None,
                "fields": preview_fields,
            },
            "summary": summary,
            "changes": changes,
        }

    def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: TeachingToolContext,
        *,
        client: TeachingAssistantClient,
        profile: ProviderProfile,
    ) -> dict[str, Any]:
        if name == "get-book-index":
            return self._book_index(arguments)
        if name == "read-book":
            return self._read_book(arguments)
        if name == "read-data":
            scoped = {
                **arguments,
                "_pure_mode": context.pure_mode,
                "_active_dataset_id": context.dataset_id,
            }
            return self._read_data(scoped)
        if name == "compress-record":
            result = self.maybe_compress(profile, client, force=True)
            return result or {"status": "没有可压缩的较早记录"}
        if name in {"create-learning-list", "write-data"}:
            if context.pending_action is not None:
                raise ValueError("已有一项操作正在等待确认")
            action = (
                self._preview_create_learning_list(arguments, context)
                if name == "create-learning-list"
                else self._preview_write_data(arguments, context)
            )
            action["id"] = f"pending_{uuid.uuid4().hex[:12]}"
            context.pending_action = action
            return {
                "status": "pending_confirmation",
                "message": "操作预览已生成，必须由用户确认后才能执行。",
                "preview": action,
            }
        raise ValueError(f"未知 AI 工具：{name}")

    def _execute_create_learning_list(
        self,
        arguments: dict[str, Any],
        profile: ProviderProfile,
    ) -> dict[str, Any]:
        mode = str(arguments["mode"])
        standard_material_id = str(arguments.get("standard_material_id") or "")
        standard_document_id = str(arguments.get("standard_document_id") or "")
        if not standard_material_id or not standard_document_id:
            raise ValueError("创建学习内容清单需要课程标准材料与文档")
        textbook_ids = [
            str(value) for value in arguments.get("textbook_material_ids", [])
        ]
        textbook_document_ids = [
            str(value)
            for value in arguments.get("textbook_document_ids", [])
        ]
        if not textbook_document_ids:
            for material_id in textbook_ids:
                textbook_document_ids.extend(
                    str(item["id"])
                    for item in self.outline.documents(material_id)
                )
        dataset_id = str(arguments.get("dataset_id") or "") or None
        scope = self.outline.prepare_two_stage_scope(
            mode=mode,
            standard_material_id=standard_material_id,
            standard_document_id=standard_document_id,
            textbook_material_ids=textbook_ids,
            textbook_document_ids=textbook_document_ids,
            dataset_id=dataset_id,
            theme_codes=[str(value) for value in arguments.get("theme_codes", [])],
        )
        selection = self.outline.select_materials(scope=scope, profile=profile)
        result = self.outline.generate_from_selection(
            scope=scope,
            selection=selection,
            specific_requirements=str(
                arguments.get("specific_requirements") or ""
            ),
            profile=profile,
            name=str(arguments.get("name") or "") or None,
        )
        return {
            "status": "created",
            "run_id": result["run"]["id"],
            "run_name": result["run"]["name"],
            "item_count": len(result["items"]),
            "relation_count": len(result["relations"]),
        }

    def execute_pending_action(
        self,
        action: dict[str, Any],
        profile: ProviderProfile,
    ) -> dict[str, Any]:
        tool = action.get("tool")
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("待确认操作缺少参数")
        if tool == "create-learning-list":
            return self._execute_create_learning_list(arguments, profile)
        if tool == "write-data":
            dataset_id = str(arguments.get("dataset_id") or "")
            data_type = str(arguments.get("data_type") or "")
            values = arguments.get("fields") or {}
            if arguments.get("action") == "create":
                row_id = self.learning.create_record(dataset_id, data_type, values)
            else:
                row_id = int(arguments["row_id"])
                self.learning.update_record(
                    dataset_id,
                    data_type,
                    row_id,
                    values,
                    allow_identity_change=True,
                )
            return {
                "status": "written",
                "dataset_id": dataset_id,
                "data_type": data_type,
                "row_id": row_id,
            }
        raise ValueError("未知待确认操作")

    # ------------------------------------------------------------------
    # agent loop
    # ------------------------------------------------------------------
    def _system_prompt(
        self,
        context: TeachingToolContext,
        dataset_name: str | None,
    ) -> str:
        catalog = self._material_catalog()
        temporary_materials = [
            {
                "name": item["name"],
                "content": item["text"],
            }
            for item in context.temporary_materials
        ]
        return f"""
你是 TailorLearn 的教学建议智能体。用中文与教师对话，给出可执行、克制、
有依据的教学建议。不能泄露任何 API Key；上下文中不会提供 API Key。

当前模式：{"纯净模式" if context.pure_mode else "普通模式"}
当前学情数据集：{dataset_name or "未选择"}

规则：
1. 现有课本、课程标准与全部目录已随上下文发送。需要正文时调用
   get-book-index 定位章节，再调用 read-book 读取。
2. 普通模式可调用 read-data 获取学情数据、班级数据、作答正文和学习内容清单。
3. 纯净模式不得读取或推断学生、班级、作答、诊断和学习主题数据；
   如需按主题建议，必须使用用户在问题中手动写出的主题。
4. 纯净模式可以读取已有学习内容清单，也可以创建课本模式学习内容清单。
5. create-learning-list 与 write-data 只是提交确认预览，不会立即写入；
   工具返回 pending_confirmation 时，告知用户等待确认，不得声称已经完成。
6. write-data 只能新增或修改，不能删除。
7. 班级当前由学生的 class_code 归属派生；学习主题与指标存储在 requirement 数据中。
8. 教学建议不是审核草稿，直接以清晰的 Markdown 回答。
9. 不要输出思考过程或工具内部推理；可以用简短步骤说明使用了哪些资料。

材料目录：
{json.dumps(catalog, ensure_ascii=False)}

临时素材（仅当前会话，不写入数据库）：
{json.dumps(temporary_materials, ensure_ascii=False)}
""".strip()

    def run_turn(
        self,
        *,
        prompt: str,
        profile: ProviderProfile,
        client: TeachingAssistantClient,
        dataset_id: str | None,
        pure_mode: bool,
        temporary_materials: list[dict[str, Any]],
    ) -> TeachingTurnResult:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("问题不能为空")
        self.append_message("user", prompt)
        dataset = self.learning.get_dataset(dataset_id) if dataset_id else None
        context = TeachingToolContext(
            dataset_id=dataset_id,
            pure_mode=pure_mode,
            temporary_materials=temporary_materials,
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._system_prompt(
                    context,
                    str(dataset["name"]) if dataset else None,
                ),
            },
            *self._model_history(),
        ]
        final_content = ""
        for _ in range(MAX_TOOL_ROUNDS):
            completion = client.complete(
                messages,
                self.tool_definitions(),
            )
            if completion.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": completion.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(
                                        call.arguments,
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                            for call in completion.tool_calls
                        ],
                    }
                )
                for call in completion.tool_calls:
                    try:
                        output = self.execute_tool(
                            call.name,
                            call.arguments,
                            context,
                            client=client,
                            profile=profile,
                        )
                        tool_content = json.dumps(
                            output,
                            ensure_ascii=False,
                        )
                    except Exception as exc:
                        tool_content = json.dumps(
                            {"error": str(exc)},
                            ensure_ascii=False,
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": tool_content,
                        }
                    )
                continue
            final_content = (completion.content or "").strip()
            if final_content:
                break
        if not final_content:
            final_content = "本次请求未得到有效回复，请稍后重试。"
        message = self.append_message("assistant", final_content)
        compression: dict[str, Any] | None = None
        try:
            compression = self.maybe_compress(profile, client)
        except Exception:
            compression = None
        return TeachingTurnResult(
            message=message,
            pending_action=context.pending_action,
            compression=compression,
        )
