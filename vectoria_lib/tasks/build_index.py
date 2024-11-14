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
from vectoria_lib.rag.preprocessing.pipeline.preprocessing_pipeline_builder import PreprocessingPipelineBuilder
from vectoria_lib.rag.vector_store.vectore_store_builder import VectorStoreBuilder

def build_index(
    **kwargs: dict
) -> tuple[Path, FaissVectorStore]:
    config = Config()

    logger = logging.getLogger("tasks")

    start_time = time.time()
    docs = PreprocessingPipelineBuilder().build_pipeline().run(
                Path(kwargs["input_docs_dir"])
            )
    logger.info("Created %d documents from %s in %.2f seconds", len(docs), kwargs['input_docs_dir'], time.time() - start_time)
        
    start_time = time.perf_counter()

    fvs = VectorStoreBuilder().build(
        config.get("vector_store"),
        index_path = None
    )         
    
    fvs.make_index(docs)
    logger.debug("Creation of FAISS index (.from_documents) took %.2f seconds", time.perf_counter() - start_time)

    start_time = time.perf_counter()
    pkl_path = fvs.dump_to_disk(kwargs["output_dir"])
    logger.info("Index pkl dumped at: %s took %.2f seconds", pkl_path, time.perf_counter() - start_time)

    return pkl_path, fvs

# TODO: AL MOMENTO ABBIAMO SOLO LA FUNZIONE CHE GENERA UN INDEX A PARTIRE DAI DOCS
# DOBBIAMO IMPLEMENTARE LA FUNZIONE CHE FA LA DELETION E UPDATE