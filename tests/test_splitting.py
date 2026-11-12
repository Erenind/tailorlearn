from __future__ import annotations

from tailorlearn.splitting import split_pages


def test_split_uses_headings_and_ignores_running_headers():
    pages = [
        {
            "page_number": 1,
            "final_text": (
                "第十一章 不等式与不等式组\n"
                "本章说明。\n"
                "11.1 不等式\n"
                "第一节内容。"
            ),
        },
        {
            "page_number": 2,
            "final_text": (
                "第十一章 不等式与不等式组\n"
                "11.1.1 不等式及其解集\n"
                "问题内容。\n"
                "3.5 dm\n"
                "0.32 0t 标准煤"
            ),
        },
    ]
    sections = split_pages(document_id="doc_test", pages=pages)
    titles = [section.title for section in sections]
    assert titles == [
        "第十一章 不等式与不等式组",
        "11.1 不等式",
        "11.1.1 不等式及其解集",
    ]
    assert sections[2].parent_id == sections[1].id
    assert "第十一章 不等式与不等式组" not in sections[2].content
    assert "3.5 dm" in sections[2].content


def test_split_falls_back_to_page_sections_without_headings():
    pages = [
        {"page_number": 1, "final_text": "这是第一页的有效内容。"},
        {"page_number": 2, "final_text": "这是第二页的有效内容。"},
    ]
    sections = split_pages(document_id="doc_test", pages=pages)
    assert [section.title for section in sections] == ["第 1 页", "第 2 页"]
    assert all(section.split_method == "page-fallback" for section in sections)


def test_toc_continuation_and_table_caption_are_not_sections():
    pages = [
        {"page_number": 1, "final_text": "目录\n一、课程性质\n…… 1\n二、课程理念\n2"},
        {"page_number": 2, "final_text": "（一）教学建议\n84\n（二）评价建议\n89\n附录\n99"},
        {
            "page_number": 3,
            "final_text": "一、课程性质\n正文第一段。\n表8 中的数据。\n25.2 元，小一些的25元。",
        },
    ]
    sections = split_pages(document_id="doc_test", pages=pages)
    assert "目录" in [section.title for section in sections]
    assert "一、 课程性质" in [section.title for section in sections]
    assert not any(section.title.startswith("表8") for section in sections)
    assert not any(section.title.startswith("25.2") for section in sections)
