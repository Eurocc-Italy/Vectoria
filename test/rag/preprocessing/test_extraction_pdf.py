from langchain.docstore.document import Document

from vectoria_lib.ingestion.extraction_pdf import extract_text_from_pdf_file
from vectoria_lib.common.paths import TEST_DIR

def test_extract_text_from_pdf_file(config):

    docs: list[Document] = extract_text_from_pdf_file(
        TEST_DIR / "data/pdf/1.pdf",
        dump_doc_structure_on_file=False,
        regexes_for_metadata_extraction=[]
    )

    assert isinstance(docs, list)
    assert isinstance(docs[0], Document)
    assert len(docs) == 1
    assert "arXiv:1906.02243v1" in docs[0].page_content
    assert set(docs[0].metadata.keys()) == set(["paragraph_name", "paragraph_number", "doc_file_name"])
