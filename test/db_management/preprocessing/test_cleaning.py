import pytest
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.db_management.preprocessing.cleaning import replace_ligatures

from langchain.docstore.document import Document

def test_clean_ligatures(data_dir):
    with open(data_dir / "raw" / "ligatures.txt", "r") as f:
        doc = Document(page_content=f.read(), metadata={})
    cleaned_doc = replace_ligatures(doc)
    for l in cleaned_doc.page_content.splitlines():
        assert len(l) > 1

