#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import time
import logging
from pathlib import Path
import multiprocessing as mp

from vectoria_lib.common.io.folder_reader import get_files_in_folder
from vectoria_lib.common.config import Config
# ----------------------------------------------------------------------------------------------------------------------
# DO NOT REMOVE THESE IMPORTS, they are get from global namespace
from vectoria_lib.rag.preprocessing.extraction_pdf import extract_text_from_pdf_file
from vectoria_lib.rag.preprocessing.extraction_docx import extract_text_from_docx_file
from vectoria_lib.rag.preprocessing.utils import seek_and_replace
from vectoria_lib.rag.preprocessing.chunking import recursive_character_text_splitter
# ----------------------------------------------------------------------------------------------------------------------from functools import partial
from langchain_core.runnables import RunnableLambda


class PreprocessingPipeline:

    logger = logging.getLogger('db_management')
    @staticmethod
    def get_extraction_fn(extraction_config: dict):
        doc_format = extraction_config.pop("format")
        if doc_format == "pdf":
            return extract_text_from_pdf_file
        elif doc_format == "docx":
            return extract_text_from_docx_file
        else:
            raise ValueError(f"Unsupported format: {doc_format}")

    @staticmethod
    def build_pipeline():
        data_ingestion_config = Config().get("data_ingestion")
        data_ingestion_steps_config = data_ingestion_config.keys()

        # Move these checks to the config validation
        if "extraction" not in data_ingestion_steps_config:
            raise ValueError("extraction key is mandatory in data ingestion pipeline")
        if "regexes_for_replacement" not in data_ingestion_steps_config:
            raise ValueError("regexes_for_replacement key is mandatory in data ingestion pipeline")
        if "chunking" not in data_ingestion_steps_config:
            raise ValueError("chunking key is mandatory in data ingestion pipeline")

        extraction_fn = PreprocessingPipeline.get_extraction_fn(data_ingestion_config["extraction"])

        chain = RunnableLambda(extraction_fn).bind(**data_ingestion_config["extraction"])
        
        chain = chain | RunnableLambda(
            seek_and_replace
        ).bind(regex_configs=data_ingestion_config["regexes_for_replacement"]).map()

        chain = chain | RunnableLambda(
            recursive_character_text_splitter
        ).bind(**data_ingestion_config["chunking"]).map()

        try:
            PreprocessingPipeline.logger.info("%s", chain.get_graph().draw_ascii())
        except Exception as e:
            PreprocessingPipeline.logger.warning("Failed to print graph: %s", e)

        return PreprocessingPipeline(chain=chain, multiprocessing=data_ingestion_config["multiprocessing"])
    
    def __init__(self, chain, multiprocessing):
        self.logger = logging.getLogger('db_management')
        self.chain = chain
        self.multiprocessing = multiprocessing

    def run(self, input_docs: Path):
        self.logger.info("Running preprocessing pipeline")
        files = get_files_in_folder(input_docs)
        self.logger.info("Found %d files", len(files))

        if len(files) == 0:
            raise ValueError(f"No files found in the input folder {input_docs}")

        if self.multiprocessing:
            self.logger.info("Running preprocessing pipeline with multiprocessing")
            with mp.Pool(processes=mp.cpu_count()) as pool:
                preprocessed_docs = pool.map(self.run_on_file, files)
        else:        
            self.logger.info("Running preprocessing pipeline without multiprocessing")
            preprocessed_docs = [self.run_on_file(file) for file in files]

        # Flatten the list of lists of Document objects:
        # - several docs per input files
        # - several paragraphs per doc
        # - several chunks per paragraph
        preprocessed_docs = [doc for document_chunks in preprocessed_docs for document_chunk in document_chunks for doc in document_chunk]
        self.logger.info("Total number of chunks: %d", len(preprocessed_docs))
        
        return preprocessed_docs

    def run_on_file(self, file_path: Path):
        start_time = time.perf_counter()
        output = self.chain.invoke(file_path)
        self.logger.debug("Processing file %s took %.2f seconds", file_path, time.perf_counter() - start_time)
        
        return output
