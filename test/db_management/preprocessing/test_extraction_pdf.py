import pytest
from pathlib import Path
from langchain.docstore.document import Document

from vectoria_lib.db_management.preprocessing.extraction_pdf import extract_text_from_pdf_file
from vectoria_lib.common.paths import TEST_DIR


def test_extract_text_from_pdf_file():
    doc: list[Document] = extract_text_from_pdf_file(TEST_DIR / "data/pdf/1.pdf", filter_paragraphs=[])

    assert isinstance(doc, list)
    assert isinstance(doc[0], Document)
    assert len(doc) == 1
    assert "arXiv:1906.02243v1" in doc[0].page_content
    assert "source" in doc[0].metadata
    assert "name" in doc[0].metadata
    assert "level" in doc[0].metadata
    assert "id" in doc[0].metadata
