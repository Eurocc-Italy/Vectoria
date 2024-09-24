import time
import argparse
import logging
from pathlib import Path

from vectoria_lib.common.config import Config
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline_builder import PreprocessingPipelineBuilder

def create_and_write_index(
    args: argparse.Namespace
) -> tuple[Path, FaissVectorStore]:
    
    logger = logging.getLogger("tasks")

    start_time = time.time()
    docs = PreprocessingPipelineBuilder().build_pipeline(
        Config(args.config)
    ).run(
        Path(args.input_docs_dir)
    )
    logger.info(f"Created {len(docs)} documents from {args.input_docs_dir} in {time.time() - start_time:.2f} seconds")

        
    start_time = time.time()
    fvs = FaissVectorStore(args.hf_embedder_model_name).make_index(docs)
    logger.info(f"Created index in {time.time() - start_time:.2f} seconds")

    return fvs.dump_to_pickle(args.output_index_dir), fvs


# TODO: AL MOMENTO ABBIAMO SOLO LA FUNZIONE CHE GENERA UN INDEX A PARTIRE DAI DOCS
# DOBBIAMO IMPLEMENTARE LA FUNZIONE CHE FA LA DELETION E UPDATE