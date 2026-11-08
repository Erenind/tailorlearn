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
