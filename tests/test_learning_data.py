from __future__ import annotations

import csv
import io
from pathlib import Path

from tailorlearn.db import Repository
from tailorlearn.learning_data import LearningDataService


def _service(tmp_path: Path) -> tuple[LearningDataService, str]:
    repository = Repository(tmp_path / "learning.sqlite3")
    repository.initialize()
    service = LearningDataService(repository, tmp_path)
    dataset = service.create_dataset("测试学情数据")
    return service, dataset["id"]


def _import(
    service: LearningDataService,
    dataset_id: str,
    data_type: str,
    filename: str,
    content: bytes,
    *,
    mapping: dict[str, str] | None = None,
    remember_mapping: bool = False,
):
    table = service.read_csv(content)
    selected = mapping or service.suggest_mapping(
        dataset_id,
        data_type,
        table.headers,
    )
    preview = service.preview_import(
        dataset_id,
        data_type,
        filename,
        content,
        selected,
    )
    assert not preview.errors, preview.errors
    return service.execute_import(
        preview,
        remember_mapping=remember_mapping,
    )


def _csv(headers: list[str], rows: list[list[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def test_student_import_uses_fields_not_filename_and_round_trips(tmp_path):
    service, dataset_id = _service(tmp_path)
    content = _csv(
        ["学生编号", "模拟班级编号", "年级", "备注"],
        [["D01", "DC01", "7", "扩展信息"]],
    )

    result = _import(
        service,
        dataset_id,
        "student",
        "完全不同的文件名.csv",
        content,
    )

    assert result.inserted == 1
    rows = service.list_records(dataset_id, "student")
    assert rows[0]["student_code"] == "D01"
    assert rows[0]["grade"] == 7
    exported, _ = service.export_csv(dataset_id, "student")
    assert exported.startswith(b"\xef\xbb\xbf")
    text = exported.decode("utf-8-sig")
    exported_rows = list(csv.DictReader(io.StringIO(text)))
    assert exported_rows[0]["学生编号"] == "D01"
    assert exported_rows[0]["备注"] == "扩展信息"

    repeated = service.preview_import(
        dataset_id,
        "student",
        "renamed-again.csv",
        content,
        service.suggest_mapping(
            dataset_id,
            "student",
            service.read_csv(content).headers,
        ),
    )
    assert repeated.counts["skip"] == 1
    assert repeated.counts["insert"] == 0


def test_manual_mapping_can_be_remembered(tmp_path):
    service, dataset_id = _service(tmp_path)
    content = _csv(
        ["学生", "班级", "年级"],
        [["D01", "DC01", "7"]],
    )
    mapping = {
        "学生": "student_code",
        "班级": "class_code",
        "年级": "grade",
    }
    _import(
        service,
        dataset_id,
        "student",
        "映射.csv",
        content,
        mapping=mapping,
        remember_mapping=True,
    )

    suggested = service.suggest_mapping(
        dataset_id,
        "student",
        ["学生", "班级", "年级"],
    )
    assert suggested == mapping


def test_reimport_does_not_overwrite_manual_edits_by_default(tmp_path):
    service, dataset_id = _service(tmp_path)
    content = _csv(
        ["学生编号", "模拟班级编号", "年级"],
        [["D01", "DC01", "7"]],
    )
    _import(service, dataset_id, "student", "students.csv", content)
    row = service.list_records(dataset_id, "student")[0]
    service.update_record(
        dataset_id,
        "student",
        row["row_id"],
        {"class_code": "MANUAL"},
    )
    mapping = service.suggest_mapping(
        dataset_id,
        "student",
        ["学生编号", "模拟班级编号", "年级"],
    )

    preview = service.preview_import(
        dataset_id,
        "student",
        "students.csv",
        content,
        mapping,
        overwrite_manual=False,
    )
    assert preview.counts["manual_conflict"] == 1
    assert preview.rows[0]["_action"] == "manual_conflict"
    result = service.execute_import(preview)
    assert result.conflicts == 1
    assert service.list_records(dataset_id, "student")[0]["class_code"] == "MANUAL"

    overwrite = service.preview_import(
        dataset_id,
        "student",
        "students.csv",
        content,
        mapping,
        overwrite_manual=True,
    )
    assert overwrite.rows[0]["_action"] == "update"
    service.execute_import(overwrite)
    assert service.list_records(dataset_id, "student")[0]["class_code"] == "DC01"


def test_answer_revisions_are_preserved_and_latest_is_optional(tmp_path):
    service, dataset_id = _service(tmp_path)
    content = _csv(
        [
            "记录编号",
            "学生编号",
            "材料编号",
            "主题编号",
            "节点编号",
            "作答事件编号",
            "修订版本",
            "模拟提交时间",
            "评分状态",
            "得分",
            "满分",
            "学生最终作答",
            "解答过程",
            "评分点得分",
        ],
        [
            [
                "DR001",
                "D01",
                "Q001",
                "T01",
                "N01",
                "D01-N01-Q001",
                "1",
                "2026-01-20T08:00:00+08:00",
                "待评分",
                "",
                "1",
                "",
                "",
                "",
            ],
            [
                "DR002",
                "D01",
                "Q001",
                "T01",
                "N01",
                "D01-N01-Q001",
                "2",
                "2026-01-21T08:00:00+08:00",
                "已评分",
                "1",
                "1",
                "答案",
                "",
                "Q001P1=1",
            ],
        ],
    )
    _import(service, dataset_id, "answer", "answers.csv", content)

    all_rows = service.list_records(dataset_id, "answer")
    latest = service.list_records(dataset_id, "answer", latest_answers=True)
    assert len(all_rows) == 2
    assert len(latest) == 1
    assert latest[0]["record_code"] == "DR002"


def test_identity_change_and_cascade_delete_keep_relations(tmp_path):
    service, dataset_id = _service(tmp_path)
    students = _csv(
        ["学生编号", "模拟班级编号", "年级"],
        [["D01", "DC01", "7"]],
    )
    materials = _csv(
        [
            "材料编号",
            "主题编号",
            "材料类型",
            "题型",
            "题干或内容",
            "选项",
            "参考答案",
            "参考解析",
            "满分",
            "评分点",
            "指标编号",
            "来源或资料文件",
        ],
        [["Q001", "T01", "题目", "填空题", "1+1", "", "2", "", "1", "", "", "测试"]],
    )
    nodes = _csv(
        ["节点编号", "模拟测验日期", "已教及测量主题", "说明", "可用题目编号"],
        [["N01", "2026-01-20", "T01", "测试节点", "Q001"]],
    )
    answers = _csv(
        [
            "记录编号",
            "学生编号",
            "材料编号",
            "主题编号",
            "节点编号",
            "作答事件编号",
            "修订版本",
            "模拟提交时间",
            "评分状态",
            "得分",
            "满分",
            "学生最终作答",
            "解答过程",
            "评分点得分",
        ],
        [
            [
                "DR001",
                "D01",
                "Q001",
                "T01",
                "N01",
                "D01-N01-Q001",
                "1",
                "2026-01-20T08:00:00+08:00",
                "已评分",
                "1",
                "1",
                "2",
                "",
                "",
            ]
        ],
    )
    _import(service, dataset_id, "student", "students.csv", students)
    _import(service, dataset_id, "material", "materials.csv", materials)
    _import(service, dataset_id, "node", "nodes.csv", nodes)
    _import(service, dataset_id, "answer", "answers.csv", answers)

    student = service.list_records(dataset_id, "student")[0]
    service.update_record(
        dataset_id,
        "student",
        student["row_id"],
        {"student_code": "S01"},
        allow_identity_change=True,
    )
    answer = service.list_records(dataset_id, "answer")[0]
    assert answer["student_code"] == "S01"

    deleted = service.soft_delete_record(
        dataset_id,
        "student",
        student["row_id"],
        cascade=True,
    )
    assert deleted == 2
    assert service.list_records(dataset_id, "answer") == []
    service.restore_record(
        dataset_id,
        "student",
        student["row_id"],
        cascade=True,
    )
    assert service.list_records(dataset_id, "student")[0]["student_code"] == "S01"
    assert service.list_records(dataset_id, "answer")[0]["student_code"] == "S01"


def test_quality_report_finds_scores_and_missing_references(tmp_path):
    service, dataset_id = _service(tmp_path)
    service.create_record(
        dataset_id,
        "answer",
        {
            "record_code": "DR001",
            "student_code": "MISSING",
            "material_code": "MISSING",
            "theme_code": "T99",
            "node_code": "N99",
            "answer_event_code": "EVENT",
            "revision": 1,
            "submitted_at": "2026-01-20T08:00:00+08:00",
            "grading_status": "已评分",
            "score": 4,
            "full_score": 3,
        },
    )
    report = service.quality_report(dataset_id)
    codes = {issue["问题代码"] for issue in report["issues"]}
    assert "missing_reference" in codes
    assert "score_out_of_range" in codes
