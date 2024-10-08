import logging
from pathlib import Path
import multiprocessing as mp
from functools import partial

from langchain_core.runnables import chain, RunnableConfig
from langchain.docstore.document import Document

from vectoria_lib.io.folder_reader import get_files_in_folder
from vectoria_lib.common.config import Config

class PreprocessingPipelineExecutor:
    """
    It builds a langchain based preprocessing pipeline to process a set of documents.
    https://python.langchain.com/docs/how_to/lcel_cheatsheet
    """
    def __init__(self, pp_chain):
        self.logger = logging.getLogger('db_management')
        self.chain = pp_chain

    def run(self, input_docs: Path):
        self.logger.info("Running preprocessing pipeline")
        files = get_files_in_folder(input_docs)
        self.logger.info("Found %d files", len(files))

        if Config().get("pp_multiprocessing"):
            self.logger.info("Running preprocessing pipeline with multiprocessing")
            with mp.Pool(processes=mp.cpu_count()) as pool:
                preprocessed_docs = pool.map(self.run_on_file, files)
        else:        
            self.logger.info("Running preprocessing pipeline without multiprocessing")
            preprocessed_docs = [self.run_on_file(file) for file in files]
        
        # at this point we have a list of lists of Document objects (one sublist per file)
        # and we need to flatten the list

        preprocessed_docs = [doc for document_chunks in preprocessed_docs for paragraph_chunks in document_chunks for doc in paragraph_chunks]
        
        self.logger.info("Total number of chunks: %d", len(preprocessed_docs))
        
        return preprocessed_docs

    def run_on_file(self, file_path: Path):
        self.logger.info("Processing file %s", file_path)
        return self.chain.invoke(file_path)
