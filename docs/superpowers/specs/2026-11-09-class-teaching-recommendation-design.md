# 班级教学建议闭环设计

## 目标

在保留现有学生诊断能力的前提下，新增独立的教师端“教学建议”模块：以可复核的班级证据包为唯一输入，生成可编辑、可审核、会随数据更新失效的 AI 教学建议，并导出脱敏典型案例。

## 范围

本轮实现 P1：班级教学建议（A）、班级分层教学（E1）、测验蓝图与覆盖缺口（E3）、典型案例导出（E5）。

不实现自由对话、学生端功能、自动向学生发送任务、AI 自动采纳建议、真实 API 的批量压测，或把数据变化解释成教学干预的因果结果。

## 不可变的数据与 AI 边界

1. 仅统计未删除的最新修订、已评分、分值有效且关联完整的作答。
2. 待评分、未作答、异常或关联缺失的数据只作为数据状态，绝不等同于未掌握。
3. 班级主题仅在至少 5 名学生和 10 条有效记录时为证据充分；不足时不形成正向或需关注的表现结论。
4. 学生主题仅在至少 3 条有效记录时为证据充分。
5. 班级分层仅使用相同班级、节点、主题下的充分证据：
   - `observed_higher`：学生主题证据充分，且已观察得分率不低于班级该主题已观察得分率；
   - `needs_attention`：学生主题证据充分，且已观察得分率低于班级该主题已观察得分率至少 20 个百分点；
   - `observed_other`：学生主题证据充分但不属于前两组；
   - `needs_observation`：学生主题证据不足或班级主题证据不足。
6. 所有文案以“已观察表现”“需关注”“待补充观察”描述，禁止使用“未掌握”“掌握不足”“能力下降”。
7. AI 不重新计算统计、不推断未提供事实；每条观察引用已有主题和指标，材料建议仅引用当前节点可用、学生尚未作答的材料编号。

## 架构

新增 `TeachingRecommendationService`，与 `StudentDiagnosisService` 并列，不复用其任务表或诊断表。该服务依赖 `AnalyticsService`：

1. 计算班级主题统计、学生主题统计、主题—指标—材料覆盖情况、分层名单和候选补充材料；
2. 组合为脱敏班级证据包；
3. 验证 AI 返回的结构化草稿是否只引用证据包允许的主题、指标、材料与分层；
4. 保存草稿、审核结果、证据包、模型元数据、原始响应及变化日志截点；
5. 学情数据变化后，将仍可展示的旧建议标记为 `stale`。

页面层只调用服务并展示返回数据。`app.py` 新增侧边栏“教学建议”入口；该页面不修改学生诊断页面。

## 数据模型

### 班级证据包

`build_evidence_pack(dataset_id, class_code, node_code)` 返回：

```json
{
  "dataset_id": "...",
  "class_code": "DC01",
  "node_code": "N01",
  "class_themes": [{"theme_code": "T01", "score_rate": 0.72, "scored_records": 12, "student_count": 6, "evidence_status": "sufficient"}],
  "assessment_blueprint": [{"theme_code": "T01", "indicator_code": "T01I1", "available_material_codes": ["Q001"], "valid_answer_count": 5, "pending_count": 1, "unanswered_count": 0, "coverage_status": "observed"}],
  "groups": [{"theme_code": "T01", "group": "needs_attention", "student_codes": ["D01"], "reason": "..."}],
  "candidate_materials": [{"theme_code": "T01", "indicator_code": "T01I2", "material_code": "Q005", "eligible_student_codes": ["D01", "D02"]}]
}
```

证据包不包含姓名、原始作答文本、解析过程、整库数据或 API Key。

### AI 草稿

新增 Pydantic 模型 `TeachingRecommendationDraft`：

```json
{
  "summary": "谨慎的班级观察摘要",
  "observations": [{"theme_code": "T01", "statement": "...", "indicator_codes": ["T01I1"]}],
  "information_gaps": [{"theme_code": "T01", "indicator_code": "T01I2", "statement": "..."}],
  "group_activities": [{"theme_code": "T01", "group": "needs_attention", "activity": "...", "student_codes": ["D01"]}],
  "material_suggestions": [{"theme_code": "T01", "indicator_code": "T01I2", "material_code": "Q005", "student_codes": ["D01", "D02"], "reason": "..."}],
  "teacher_actions": ["..."],
  "post_lesson_checks": ["..."]
}
```

校验器拒绝主题、指标、材料、分组或材料适用学生名单外引用；拒绝包含禁止结论的所有字段内容。

### SQLite

新增 `ai_teaching_recommendations` 表，每行对应一个“数据集＋班级＋节点”的草稿，保存：

- `id`、`dataset_id`、`class_code`、`node_code`；
- `change_log_cutoff`、`evidence_pack_json`、`response_json`、`raw_response`；
- 厂商、base URL、模型与提示词版本（均不含密钥）；
- `status`：`needs_review`、`accepted`、`rejected`、`stale`；
- 教师修订摘要、审核备注、创建/更新/审核时间。

为 `dataset_id, class_code, node_code, created_at` 建索引。初始化将 schema 版本从 6 更新为 7，保持对既有数据库的兼容创建。

## AI 调用与审核

`OpenAICompatibleClient.draft_teaching_recommendation(evidence_pack)` 提示模型只输出上述 JSON。AI 调用失败、JSON 解析失败或证据校验失败时不落库草稿，页面给出明确错误并可重新生成。

教师可编辑摘要、填写审核备注，并选择采纳或拒绝。采纳后的教师修订与原始 AI 响应同时保留。数据变化后，`refresh_stale_recommendations(dataset_id)` 将 `needs_review` 和 `accepted` 建议标记为 `stale`；拒绝记录保持历史状态。

## 页面与导出

“教学建议”页面依序展示：

1. 数据集、班级、测验节点选择器；
2. 班级主题统计与证据状态；
3. 测验蓝图：主题、指标、可用题目、有效/待评分/未作答记录及覆盖状态；
4. 每主题的分层名单；当证据不足时仅展示待补充观察；
5. 生成或查看 AI 草稿、证据包、审核控件和失效提示；
6. JSON、CSV、Markdown 三种脱敏案例导出。

Markdown 导出包含范围、统计规则、班级主题结果、测验蓝图、分层、AI 草稿和审核信息；JSON 导出完整证据包与草稿；CSV 导出主题统计、蓝图和分层三份表格。导出内容不含原始作答、API Key 和 SQLite 文件。

## 测试与验收

测试先行覆盖：

1. 班级证据包只包含允许字段，不泄露 `final_answer`；
2. 班级主题或学生主题证据不足时仅产生 `needs_observation`，不生成表现/关注分层；
3. 蓝图区分有效、待评分、未作答和无题目覆盖；
4. AI 草稿引用范围外主题、指标、材料、学生或带禁止结论时被拒绝；
5. 合规草稿可保存、教师可采纳/拒绝，数据变化后正确失效；
6. 三种导出包含应有范围与审核信息，且排除敏感字段；
7. 页面测试确认“教学建议”入口和核心提示存在；
8. 全量回归测试通过，并手动在 Streamlit 页面验证无 API 的安全降级及配置 API 后的生成/审核路径。
