#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import time
import logging
from pathlib import Path

from vectoria_lib.common.config import Config
from vectoria_lib.components.vector_store.vector_store_factory import VectorStoreFactory 
from vectoria_lib.components.vector_store.vectore_store_base import VectorStoreBase
from vectoria_lib.ingestion.preprocessing_pipeline import PreprocessingPipeline
from vectoria_lib.common.io.folder_reader import get_files_in_folder

def build_index(
    **kwargs: dict
) -> tuple[Path, VectorStoreBase]:
    logger = logging.getLogger("tasks")
    files = get_files_in_folder(Path(kwargs["input_docs_dir"]))
    for file in files:
        logger.debug("Found file: %s", file)
    if len(files) == 0:
        raise ValueError(f"No files found in the input folder {kwargs['input_docs_dir']}")
    return build_index_from_files(files, Path(kwargs["output_dir"]))

def build_index_from_files(
    files: list[Path],
    output_index_dir: Path
) -> tuple[Path, VectorStoreBase]:

    config = Config()
    logger = logging.getLogger("tasks")

    start_time = time.time()
    docs = PreprocessingPipeline.build_pipeline().run(files)
    logger.info("Created %d documents from %s in %.2f seconds", len(docs), files, time.time() - start_time)
        

    start_time = time.perf_counter()
    vector_store = VectorStoreFactory.create_vector_store(**config.get("vector_store"))
    
    vector_store.make_index(docs)
    logger.debug("Index creation took %.2f seconds", time.perf_counter() - start_time)


    start_time = time.perf_counter()
    pkl_path = vector_store.dump_to_disk(output_index_dir)
    logger.info("Index pkl dumped at: %s took %.2f seconds", pkl_path, time.perf_counter() - start_time)


    return pkl_path, vector_store