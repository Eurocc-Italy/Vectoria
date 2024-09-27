import logging
from pathlib import Path
import multiprocessing as mp

from vectoria_lib.db_management.preprocessing.extraction_base import DocumentData, ExtractionBase
from vectoria_lib.db_management.preprocessing.cleaning import Cleaning
from vectoria_lib.db_management.preprocessing.chunking import Chunking
from vectoria_lib.io.folder_reader import get_files_in_folder
from vectoria_lib.common.config import Config

class PreprocessingPipeline:

    def __init__(self):
        self.logger = logging.getLogger('db_management')
        self.text_extractor: ExtractionBase = None
        self.text_cleaner: Cleaning = None
        self.chunking: Chunking = None

    def __str__(self):
        return f"PreprocessingPipeline(text_extractor={self.text_extractor}, text_cleaner={self.text_cleaner})"
    
    def set_text_extractor(self, text_extractor: ExtractionBase):
        self.text_extractor = text_extractor

    def set_text_cleaner(self, text_cleaner: Cleaning):
        self.text_cleaner = text_cleaner

    def set_chunking(self, chunking: Chunking):
        self.chunking = chunking

    def run(self, input_docs: Path):
        self.logger.info("Running preprocessing pipeline")
        files = get_files_in_folder(input_docs)
        self.logger.info("Found %d files", len(files))

        if not Config().get("multiprocessing_for_preprocessing"):
            preprocessed_docs = [self.run_on_file(file) for file in files]
                    
        with mp.Pool(processes=mp.cpu_count()) as pool:
            preprocessed_docs = pool.map(self.run_on_file, files)
        
        # at this point we have a list of lists of Document objects (one sublist per file)
        # and we need to flatten the list
        preprocessed_docs = [doc for sublist in preprocessed_docs for doc in sublist]
        
        return preprocessed_docs
    

    def run_on_file(self, path: Path):
        document_data: DocumentData = self.text_extractor.extract_text_from_file(path)
        # TODO: 
        cleaned_text = self.text_cleaner.clean_text(text)
        chunks = self.chunking.make_chunks(cleaned_text)
        self.logger.debug("Extracted %d chunks from file %s", len(chunks), path)
        return chunks