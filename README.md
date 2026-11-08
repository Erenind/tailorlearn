# TailorLearn

面向个性化学习分析的材料结构化工具。当前实现包括 AI 厂商配置、课本/课程标准
导入、本地 OCR、章节切分、AI 结构化以及批量审核。

## 启动

```bash
uv sync
uv run streamlit run src/tailorlearn/app.py
```

也可以运行：

```bash
uv run tailorlearn
```

AI 参数页支持保存多套 OpenAI-compatible Chat Completions 配置。API Key 只写入
本机 `.local/ai_profiles.json`，不读取环境变量，也不写入 `db/`。

## 学情数据

侧边栏“学情数据”页面支持按数据集分别导入、导出和维护以下 CSV：

- 课程要求
- 题目与材料
- 模拟学生
- 测验节点
- 作答记录
- 字段说明

文件名和列顺序不参与类型判断。导入前会按字段名自动匹配，必要时可人工映射并
保存别名。页面在写入前展示新增、更新、未变化、人工修改冲突和质量问题预览；
同一类型重复导入时按业务主键合并，不会静默覆盖人工修改。

作答记录保留全部修订版本，可切换“仅最新修订”；删除默认是软删除并可恢复。
“数据质量”页签会检查必填字段、关联引用、评分范围和评分状态一致性。详细存储
约定见 [db/README.md](db/README.md)。

## 离线导入

```bash
uv run tailorlearn import-pdf \
  --name "七年级数学课本切片" \
  --material-type textbook \
  --subject 数学 \
  /path/to/textbook-slice.pdf
```

成功后可在界面中审核，或使用当前厂商配置执行：

```bash
uv run tailorlearn structure --document-id <document_id>
```

`uv run tailorlearn list` 可查看材料与文档 ID。

## 数据

结构说明见 [db/README.md](db/README.md)。导入后的原始 PDF、逐页文本、章节、
单元、版本和任务索引均位于 `db/`。

## 测试

```bash
uv run pytest -q
```

具体业务规划见 [PLAN.md](PLAN.md)。
