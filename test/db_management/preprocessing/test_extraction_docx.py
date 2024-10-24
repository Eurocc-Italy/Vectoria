import pytest
from pathlib import Path
from langchain.docstore.document import Document

from vectoria_lib.db_management.preprocessing.extraction_docx import extract_text_from_docx_file
from vectoria_lib.common.paths import TEST_DIR


def test_extract_text_from_docx_file():
    doc: list[Document] = extract_text_from_docx_file(TEST_DIR / "data/docx/2.docx", filter_paragraphs=[])

    assert isinstance(doc, list)
    assert isinstance(doc[0], Document)
    assert len(doc) == 3
    assert "End first chapter" in doc[0].page_content
    assert doc[0].metadata is not None
