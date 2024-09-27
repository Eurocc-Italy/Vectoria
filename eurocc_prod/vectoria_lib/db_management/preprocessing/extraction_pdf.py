#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from .extraction_base import ExtractionBase, DocumentData

class PDFTextExtractor(ExtractionBase):

    def __init__(self):
        self.logger = logging.getLogger('db_management')
    
    def extract_text_from_file(self, file_path: Path) -> DocumentData:

        self.logger.debug(f"Extracting text from {file_path}")

        pages = PyPDFLoader(file_path).load()

        pages_list = []
        for page in pages:
            pages_list.append(page.page_content)

        pages_str = "".join(pages_list)

        self.logger.debug(f"Loaded {len(pages_str)} characters")

        return DocumentData(None, pages_str)

