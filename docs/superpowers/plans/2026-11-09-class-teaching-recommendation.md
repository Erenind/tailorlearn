# 班级教学建议闭环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于可复核的班级证据，提供分层教学建议、测验蓝图、教师审核与脱敏案例导出。

**Architecture:** 独立的 `TeachingRecommendationService` 从确定性统计构建班级证据包，再调用 OpenAI-compatible 客户端生成经校验的草稿。新页面只呈现服务数据和审核操作，草稿独立于学生诊断保存。

**Tech Stack:** Python 3.12、SQLite、Pydantic、Streamlit、Plotly、pytest、OpenAI-compatible Chat Completions。

**Spec:** `docs/superpowers/specs/2026-11-09-class-teaching-recommendation-design.md`

## Global Constraints

- 统计只使用最新、有效、已评分、关联完整的作答。
- 班级充分证据为至少 5 名学生和 10 条有效记录；学生充分证据为至少 3 条有效记录。
- 证据不足只能表示“待补充观察”，AI 与页面不得使用未掌握、掌握不足或能力下降。
- AI 只引用证据包内的主题、指标、分层学生和候选材料；不发送姓名、原始作答或密钥。
- 推荐材料必须为当前节点可用且对建议中每名目标学生均尚未作答的题目或学习卡。
- 草稿须经教师采纳或拒绝，学习数据改变后标记为 `stale`。

---

### Task 1: 数据契约与持久化

**Files:**
- Modify: `src/tailorlearn/models.py`
- Modify: `src/tailorlearn/db.py`
- Create: `tests/test_teaching_recommendations.py`

**Interfaces:** `TeachingObservation`、`TeachingInformationGap`、`TeachingGroupActivity`、`TeachingMaterialSuggestion`、`TeachingRecommendationDraft`；`ai_teaching_recommendations` 表。

- [ ] **Step 1: 写失败的模型测试**

```python
def test_teaching_draft_requires_non_blank_summary() -> None:
    with pytest.raises(ValidationError):
        TeachingRecommendationDraft.model_validate({"summary": " "})
```

- [ ] **Step 2: 确认测试因模型不存在而失败**

Run: `uv run pytest tests/test_teaching_recommendations.py::test_teaching_draft_requires_non_blank_summary -v`

- [ ] **Step 3: 实现最小 Pydantic 草稿模型**

```python
class TeachingRecommendationDraft(BaseModel):
    summary: str
    observations: list[TeachingObservation] = Field(default_factory=list)
    information_gaps: list[TeachingInformationGap] = Field(default_factory=list)
    group_activities: list[TeachingGroupActivity] = Field(default_factory=list)
    material_suggestions: list[TeachingMaterialSuggestion] = Field(default_factory=list)
    teacher_actions: list[str] = Field(default_factory=list)
    post_lesson_checks: list[str] = Field(default_factory=list)
```

Use the existing stripped non-blank summary validator.

- [ ] **Step 4: 写失败的数据库初始化测试并实现表**

```python
assert "ai_teaching_recommendations" in table_names
```

Add scope, change-log cutoff, evidence/response/raw JSON, non-secret provider metadata, prompt version, review fields and timestamps; add `idx_ai_teaching_recommendations_scope`; advance schema version to `7`.

- [ ] **Step 5: 运行任务测试；以 `feat: add teaching recommendation data contract` 提交。**

Run: `uv run pytest tests/test_teaching_recommendations.py -v`

### Task 2: 班级证据包、分层与测验蓝图

**Files:**
- Create: `src/tailorlearn/teaching_recommendations.py`
- Modify: `tests/test_teaching_recommendations.py`

**Interfaces:**

```python
class TeachingRecommendationService:
    def build_evidence_pack(self, dataset_id: str, class_code: str, node_code: str) -> dict[str, Any]: ...
    def export_case_bundle(self, dataset_id: str, class_code: str, node_code: str) -> dict[str, Any]: ...
```

- [ ] **Step 1: 写失败的充分证据分层测试**

```python
pack = service.build_evidence_pack(dataset_id, "DC01", "N01")
assert group_for(pack, "T01", "needs_attention")["student_codes"] == ["D02"]
assert group_for(pack, "T01", "needs_observation")["student_codes"] == ["D06"]
```

Seed five students with ten valid records, D02 at least 20 个百分点低于班级主题得分率，D06 少于三条有效记录。

- [ ] **Step 2: 确认失败后实现证据包**

Run: `uv run pytest tests/test_teaching_recommendations.py::test_evidence_pack_groups_only_sufficient_student_and_class_evidence -v`

Build `class_themes` from `AnalyticsService.class_theme_metrics`; create `assessment_blueprint` per requirement with `available_material_codes`, `valid_answer_count`, `pending_count`, `unanswered_count`, and `coverage_status` (`observed`/`pending_review`/`unanswered_only`/`no_available_material`). Build four groups defined by the approved spec.

- [ ] **Step 3: 写隐私与覆盖缺口测试**

```python
assert blueprint["pending_count"] == 1
assert "final_answer" not in json.dumps(pack, ensure_ascii=False)
assert blueprint["coverage_status"] == "pending_review"
```

- [ ] **Step 4: 实现候选材料与导出数据**

Candidate materials are current-node `题目`/`学习卡` only, mapped to the indicator and omitted for students with a valid answer to them. Export data contains range, evidence, recommendation and review, never raw answers or secrets.

- [ ] **Step 5: 运行测试；以 `feat: build class teaching evidence packs` 提交。**

Run: `uv run pytest tests/test_teaching_recommendations.py -v`

### Task 3: AI 草稿、校验、审核与失效

**Files:**
- Modify: `src/tailorlearn/ai.py`
- Modify: `src/tailorlearn/teaching_recommendations.py`
- Modify: `tests/test_ai_validation.py`
- Modify: `tests/test_teaching_recommendations.py`

**Interfaces:**

```python
def draft_teaching_recommendation(self, evidence_pack: dict[str, Any]) -> tuple[dict[str, Any], str]: ...
def create_recommendation(self, dataset_id: str, class_code: str, node_code: str,
                          profile: ProviderProfile, client: TeachingRecommendationClient) -> dict[str, Any]: ...
def review_recommendation(self, recommendation_id: str, *, status: str,
                          reviewed_summary: str, review_note: str) -> dict[str, Any]: ...
```

- [ ] **Step 1: 写客户端 JSON 解析失败测试**

```python
draft, _ = client.draft_teaching_recommendation({"class_code": "DC01"})
assert draft["summary"] == "谨慎摘要"
```

- [ ] **Step 2: 确认失败后实现客户端方法**

Run: `uv run pytest tests/test_ai_validation.py::test_client_parses_teaching_recommendation_json -v`

Reuse the existing JSON repair pattern, with a Chinese prompt that forbids recomputation, negative mastery claims and out-of-pack citations.

- [ ] **Step 3: 写服务拒绝越界引用和禁止结论的失败测试**

```python
with pytest.raises(ValueError, match="范围外|信息不足"):
    service.create_recommendation(dataset_id, "DC01", "N01", profile, InvalidClient())
```

- [ ] **Step 4: 实现校验、保存、审核与失效**

Validate theme/indicator/material/group/student references against the evidence pack before inserting `needs_review`, including that every `material_suggestions.student_codes` entry is eligible for that material. Permit only `accepted`/`rejected` review decisions. Mark old `needs_review` and `accepted` rows stale when `learning_change_log` advances.

- [ ] **Step 5: 运行测试；以 `feat: add reviewable class teaching recommendations` 提交。**

Run: `uv run pytest tests/test_ai_validation.py tests/test_teaching_recommendations.py -v`

### Task 4: 教学建议页面与导出

**Files:**
- Create: `src/tailorlearn/teaching_recommendations_ui.py`
- Modify: `src/tailorlearn/app.py`
- Modify: `tests/test_analytics_ui.py`

**Interfaces:**

```python
def render_teaching_recommendations_page(
    service: TeachingRecommendationService,
    learning: LearningDataService,
    profile: ProviderProfile | None,
) -> None: ...
```

- [ ] **Step 1: 写侧边栏入口失败测试**

```python
assert '"教学建议"' in Path("src/tailorlearn/app.py").read_text(encoding="utf-8")
```

- [ ] **Step 2: 确认失败后实现页面和导航**

Run: `uv run pytest tests/test_analytics_ui.py::test_application_exposes_teaching_recommendation_page -v`

Render dataset/class/node selectors, class-theme table, blueprint, groups, AI draft/review controls, stale prompt and no-profile safe guidance. Instantiate the service in `app.py`, add the sidebar item, and call renderer.

- [ ] **Step 3: 实现导出并测试**

```python
def recommendation_markdown(bundle: dict[str, Any]) -> str: ...
def recommendation_csv_sections(bundle: dict[str, Any]) -> dict[str, str]: ...
```

Add JSON, Markdown and CSV downloads; test that raw response, final answers, API keys and SQLite paths are absent.

- [ ] **Step 4: 运行测试；以 `feat: add teaching recommendation page` 提交。**

Run: `uv run pytest tests/test_analytics_ui.py tests/test_teaching_recommendations.py -v`

### Task 5: 验收

**Files:** No production-file changes expected.

- [ ] **Step 1: 运行完整回归**

Run: `uv run pytest`

- [ ] **Step 2: 检查提交范围与空白错误**

Run: `git diff --check && git status --short`

- [ ] **Step 3: 启动页面并人工检查**

Run: `uv run streamlit run src/tailorlearn/app.py --server.port 8503`

Verify the sidebar “教学建议” page: blueprint/group display, no-API safe fallback, export downloads, and—when a configured provider exists—generate/review/stale lifecycle. Keep the page open for user acceptance.
