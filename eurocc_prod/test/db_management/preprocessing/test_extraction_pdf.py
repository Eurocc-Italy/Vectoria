import pytest
from pathlib import Path
from vectoria_lib.db_management.preprocessing.extraction_pdf import PDFTextExtractor
from vectoria_lib.common.paths import TEST_DIR

@pytest.fixture
def extraction_pdf():
    return PDFTextExtractor()

def test_extract_text_from_file(extraction_pdf):
    text = extraction_pdf.extract_text_from_file(TEST_DIR / "data/pdf/1.pdf")
    assert len(text) == 24815

# def test_extract_text_from_folder(extraction_pdf):
#     text_per_doc = extraction_pdf.extract_text_from_folder(TEST_DIR / "data/pdf")
#     assert len(text_per_doc) == 2
#     assert len(text_per_doc[0]) == 24815
#     assert len(text_per_doc[1]) == 27417

# def test_extract_text_from_folder_with_limit(extraction_pdf):
#     text_per_doc = extraction_pdf.extract_text_from_folder(
#         TEST_DIR / "data/pdf",
#         limit = 1
#     )
#     assert len(text_per_doc) == 1
#     assert len(text_per_doc[0]) == 24815
