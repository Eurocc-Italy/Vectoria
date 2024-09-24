import pytest
from pathlib import Path
from vectoria_lib.db_management.preprocessing.extraction_docx import DocXTextExtractor
from vectoria_lib.common.paths import TEST_DIR

@pytest.fixture
def extraction_docx():
    return DocXTextExtractor()

def test_extract_text_from_file(extraction_docx):
    text = extraction_docx.extract_text_from_file(TEST_DIR / "data/docx/2.docx")
    assert len(text) == 65

# def test_extract_text_from_folder(extraction_docx):
#     text_per_doc = extraction_docx.extract_text_from_folder(TEST_DIR / "data/docx")
#     assert len(text_per_doc) == 2
#     assert len(text_per_doc[0]) == 65
#     assert len(text_per_doc[1]) == 904

# def test_extract_text_from_folder_with_limit(extraction_docx):
#     text_per_doc = extraction_docx.extract_text_from_folder(
#         TEST_DIR / "data/docx",
#         limit = 1
#     )
#     assert len(text_per_doc) == 1
#     assert len(text_per_doc[0]) == 65
