# `db/` 数据目录

`db/` 同时保存可查询索引、不可变原始资料和中间结果。长文本使用 JSON/JSONL
保存，SQLite 只负责查询、关联、任务状态和版本索引。

```text
db/
├── tailorlearn.sqlite3
├── learning/
│   └── <dataset_id>/
│       └── imports/<batch_id>/<original-file-name>.csv
├── materials/
│   └── <material_set_id>/
│       ├── manifest.json
│       └── <document_id>/
│           ├── source/original.pdf
│           ├── extracted/pages.jsonl
│           ├── ocr/pages.jsonl
│           ├── structure/
│           │   ├── sections.json
│           │   └── units.jsonl
│           └── runs/<run_id>/chunks/
├── exports/
└── backups/
```

约定：

- `material_set` 是一套有名称、类型和版本的材料。
- `document` 是一个 PDF 切片，由内容哈希和材料集共同生成稳定 ID。
- `section` 是章节切分块，`unit` 是 AI 或人工整理后的语义单元。
- 单元修改和 AI 重跑会生成新版本，不覆盖旧版本记录。
- 删除审核单元默认是软删除；原始 PDF、OCR 和来源引用始终保留。
- API Key 不保存在这里，而保存在 Git 忽略的项目 `.local/` 目录。
- 学生、课程要求、题目材料、测验节点、作答记录和字段说明使用独立
  `learning_*` 表，不混入 `materials/`。
- 每次学情 CSV 导入的原始文件保存在 `learning/`，批次、字段别名、质量问题和
  增删改审计均写入 SQLite。
- 学情数据先按 `learning_datasets` 隔离；同一数据集内的重复导入按业务主键合并。
- 业务编号修改不改变内部 `row_id` 关系；删除学生、材料或节点时可选择同步
  软删除关联作答，恢复时也可同步恢复。

常用表：

- `material_sets`：材料集和可编辑元数据。
- `documents`：PDF、哈希、页数和提取状态。
- `pages`：逐页文本、OCR 置信度和异常状态。
- `sections`：章节树和页码范围。
- `units` / `unit_versions`：当前单元和完整版本历史。
- `jobs` / `job_items`：导入、OCR 和 AI 任务状态。
- `learning_datasets`：学情数据集隔离边界。
- `learning_import_batches` / `learning_import_issues`：CSV 导入批次和质量提示。
- `learning_students`、`learning_requirements`、`learning_materials`、
  `learning_nodes`、`learning_answers`、`learning_field_definitions`：
  六类强类型业务数据。
- `learning_change_log`：导入和人工增删改审计。
- `learning_field_aliases`：按数据集和数据类型保存的字段映射别名。
