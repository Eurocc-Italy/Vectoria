#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

from vectoria_lib.db_management.preprocessing.document_data import  DocumentData

logger = logging.getLogger('db_management')

def extract_text_from_pdf_file(file_path: Path, filter_paragraphs=None, log_in_folder=None) -> list[Document]:
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

    return [Document(page_content=pages_str, metadata=dict(source=file_path.stem,name=file_path.name, level=0, id=0))]

