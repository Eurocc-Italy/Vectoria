import pytest
from pathlib import Path
from vectoria_lib.db_management.preprocessing.extraction_docx import DocXTextExtractor
from vectoria_lib.common.paths import TEST_DIR

@pytest.fixture
def extraction_docx():
    return DocXTextExtractor()

def test_extract_text_from_file(extraction_docx):
    text = extraction_docx.extract_text_from_file(TEST_DIR / "data/docx/2.docx")
    assert len(text) == 85
