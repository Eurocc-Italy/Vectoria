import time
import logging
from pathlib import Path

from vectoria_lib.common.config import Config
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline_builder import PreprocessingPipelineBuilder

def build_index(
    **kwargs: dict
) -> tuple[Path, FaissVectorStore]:
    
    logger = logging.getLogger("tasks")

    start_time = time.time()
    docs = PreprocessingPipelineBuilder().build_pipeline().run(
                Path(kwargs["input_docs_dir"])
            )
    logger.info("Created %d documents from %s in %.2f seconds", len(docs), kwargs['input_docs_dir'], time.time() - start_time)
        
    start_time = time.perf_counter()
    fvs = FaissVectorStore(Config().get("hf_embedder_model_name")).make_index(docs)
    logger.debug("Creation of FAISS index (.from_documents) took %.2f seconds", time.perf_counter() - start_time)

    start_time = time.perf_counter()
    pkl_path = fvs.dump_to_pickle(kwargs["output_dir"], kwargs["output_suffix"])
    logger.info("Index pkl dumped at: %s took %.2f seconds", pkl_path, time.perf_counter() - start_time)

    return pkl_path, fvs

# TODO: AL MOMENTO ABBIAMO SOLO LA FUNZIONE CHE GENERA UN INDEX A PARTIRE DAI DOCS
# DOBBIAMO IMPLEMENTARE LA FUNZIONE CHE FA LA DELETION E UPDATE