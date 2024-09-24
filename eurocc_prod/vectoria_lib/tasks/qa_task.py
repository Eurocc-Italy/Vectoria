#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#


# ----------------------------------------------------------------------------------------------
import time
import argparse
import logging
from vectoria_lib.llm.agents.qa import QAAgent
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.db_management.retriever.faiss_retriever import FaissRetriever

def create_qa_agent(
    args: argparse.Namespace,
) -> QAAgent:

    logger = logging.getLogger("tasks")

    # Load vector store
    logger.debug(f"Loading Faiss index {args.faiss_index_name}")
    vector_store = FaissVectorStore.load_from_pickle(args.faiss_index_name, args.faiss_index)
    logger.info(f"Loaded Faiss index {args.faiss_index_name} in {time.time() - start_time:.2f} seconds")

    # Create rag retriever
    logger.debug(f"Creating Faiss retriever with k={args.k}")
    retriever = FaissRetriever(search_kwargs={"k": args.k})
    retriever.set_retriever(vector_store.as_retriever())
    logger.info(f"Set Faiss retriever in {time.time() - start_time:.2f} seconds")

    # Create QA agent
    return QAAgent(retriever)