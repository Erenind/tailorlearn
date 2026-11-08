from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    for index in range(1, 4):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"Synthetic section {index}",
            fontsize=18,
        )
        page.insert_text(
            (72, 110),
            (
                "This synthetic page contains enough readable text for the "
                "text extraction path used by the importer."
            ),
            fontsize=11,
        )
    document.save(path)
    document.close()
    return path

