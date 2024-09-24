#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import os
import logging
from pathlib import Path
import multiprocessing as mp

from langchain_community.document_loaders import PyPDFLoader

from .extraction_base import ExtractionBase
from vectoria_lib.io.folder_reader import get_files_in_folder


class PDFTextExtractor(ExtractionBase):

    def __init__(self):
        self.logger = logging.getLogger('db_management')

    # def extract_text_from_folder(self, folder_path: Path, limit: int = -1) -> list[str]:

    #     files = get_files_in_folder(folder_path, limit)

    #     self.logger.debug(f"Extracting text from {len(files)} PDFs")

    #     with mp.Pool(processes=mp.cpu_count()) as pool:
    #         texts = pool.map(self.extract_text_from_file, files)

    #     return texts
    
    def extract_text_from_file(self, file_path: Path) -> str:
        """
        Extract text from a single PDF file.

        Parameters:
        - file_path (Path): The path to the PDF file from which text should be extracted.

        Returns:
        - str: A string containing all the extracted text from the file.
        """

        self.logger.debug(f"Extracting text from {file_path}")

        pages = PyPDFLoader(file_path).load()

        pages_list = []
        for page in pages:
            pages_list.append(page.page_content)

        pages_str = "".join(pages_list)

        self.logger.debug(f"Loaded {len(pages_str)} characters")
        
        return pages_str