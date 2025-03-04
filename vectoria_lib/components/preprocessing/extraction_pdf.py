#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

from vectoria_lib.components.preprocessing.document_data import  DocumentData

logger = logging.getLogger('rag')

def extract_text_from_pdf_file(
        file_path: Path,
        dump_doc_structure_on_file: bool = False,
        regexes_for_metadata_extraction: list[dict] = []
) -> list[Document]:
    """
    Extracts text from a PDF file and returns a list of Document objects.

    Each Document object contains the page content and metadata.
    The metadata must include the keys: name, level, and id.
    """
    logger.debug("Extracting text from %s", file_path)

    pages = PyPDFLoader(file_path).load()

    pages_list = []
    for page in pages:
        pages_list.append(page.page_content)

    pages_str = "".join(pages_list)

    logger.debug("Loaded %d characters", len(pages_str))

    return [
        Document(
            page_content=pages_str,
            metadata=dict(
                doc_file_name=file_path.stem,
                paragraph_name="document_name",
                paragraph_number=0
            )
        )
    ]

