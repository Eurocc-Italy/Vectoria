#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import os
import logging
from pathlib import Path
import multiprocessing as mp

from docx import Document

from .extraction_base import ExtractionBase
from vectoria_lib.io.folder_reader import get_files_in_folder


class DocXTextExtractor(ExtractionBase):
    """
    TODO: it should be a base class as well. The user should be able to choose 
    between different document parsing strategies based on the context.
    """
    def __init__(self):
        self.logger = logging.getLogger('db_management')

    # def extract_text_from_folder(self, folder_path: Path, limit: int = -1) -> list[str]:

    #     files = get_files_in_folder(folder_path, limit)
        
    #     self.logger.debug(f"Extracting text from {len(files)} PDFs")

    #     with mp.Pool(processes=mp.cpu_count()) as pool:
    #         texts = pool.map(self.extract_text_from_file, files)

    #     return texts
    
    def extract_text_from_file(self, file_path: Path) -> str:

        self.logger.debug(f"Extracting text from {file_path}")

        document = Document(file_path)
    
        pages_str = "".join([p.text for p in document.paragraphs])

        self.logger.debug(f"Loaded {len(pages_str)} characters")
        
        return pages_str