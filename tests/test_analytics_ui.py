from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from tailorlearn.db import Repository
from tailorlearn.paths import AppPaths


def test_streamlit_renders_student_and_class_analysis_navigation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = AppPaths(tmp_path)
    paths.ensure()
    Repository(paths.database_file).initialize()
    monkeypatch.setattr(AppPaths, "default", classmethod(lambda cls: paths))
    st.cache_resource.clear()
    app_file = Path(__file__).resolve().parents[1] / "src" / "tailorlearn" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=20).run()

    assert "学生分析" in at.radio(key="main_navigation").options
    assert "班级分析" in at.radio(key="main_navigation").options
    at.radio(key="main_navigation").set_value("学生分析").run()
    assert not at.exception
    assert any(item.value == "学生分析" for item in at.title)
    st.cache_resource.clear()
