#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import time
import logging
from pathlib import Path

from vectoria_lib.common.config import Config
from vectoria_lib.rag.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.rag.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from vectoria_lib.rag.vector_store.vectore_store_builder import VectorStoreBuilder
from vectoria_lib.common.io.folder_reader import get_files_in_folder

def build_index(
    **kwargs: dict
) -> tuple[Path, FaissVectorStore]:
    logger = logging.getLogger("tasks")
    files = get_files_in_folder(Path(kwargs["input_docs_dir"]))
    for file in files:
        logger.debug("Found file: %s", file)
    if len(files) == 0:
        raise ValueError(f"No files found in the input folder {kwargs['input_docs_dir']}")
    return build_index_from_files(files, Path(kwargs["output_index_dir"]))

def build_index_from_files(
    files: list[Path],
    output_index_dir: Path
) -> tuple[Path, FaissVectorStore]:
    config = Config()
    logger = logging.getLogger("tasks")

    start_time = time.time()
    docs = PreprocessingPipeline.build_pipeline().run(files)
    logger.info("Created %d documents from %s in %.2f seconds", len(docs), files, time.time() - start_time)
        

    start_time = time.perf_counter()
    vector_store = VectorStoreBuilder().build(
        config.get("vector_store"),
        index_path = None
    )         
    vector_store.make_index(docs)
    logger.debug("Creation of FAISS index (.from_documents) took %.2f seconds", time.perf_counter() - start_time)


    start_time = time.perf_counter()
    pkl_path = vector_store.dump_to_disk(output_index_dir)
    logger.info("Index pkl dumped at: %s took %.2f seconds", pkl_path, time.perf_counter() - start_time)


    return pkl_path, vector_store

# TODO: AL MOMENTO ABBIAMO SOLO LA FUNZIONE CHE GENERA UN INDEX A PARTIRE DAI DOCS
# DOBBIAMO IMPLEMENTARE LA FUNZIONE CHE FA LA DELETION E UPDATE