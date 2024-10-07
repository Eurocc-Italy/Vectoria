import time
import logging
from pathlib import Path

from vectoria_lib.llm.agents.qa import QAAgent
from vectoria_lib.common.config import Config
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.db_management.retriever.faiss_retriever import FaissRetriever
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
class AgentBuilder:

    @staticmethod
    def build_qa_agent(
        **kwargs: dict
    ) -> QAAgent:

        config = Config()
        logger = logging.getLogger("llm")
        
        # Load vector store
        start_time = time.time()
        faiss_index_path = kwargs['faiss_index_path']
        logger.debug("Loading Faiss index: %s", faiss_index_path)
        vector_store = FaissVectorStore.load_from_pickle(
            Path(faiss_index_path)
        )
        logger.info("Loaded Faiss index %s in %.2f seconds", faiss_index_path, time.time() - start_time)

        # Create rag retriever
        logger.debug("Creating Faiss retriever")
        retriever = FaissRetriever(
            vector_store,
            search_type=config.get("retriever_search_type"),
            search_kwargs={
                "k": config.get("retriever_top_k"),
                "fetch_k": config.get("retriever_fetch_k"), 
                "lambda_mult": config.get("retriever_lambda_mult")
            }
        )
        
        logger.info("Set Faiss retriever in %.2f seconds", time.time() - start_time)

        # Create QA agent
        return QAAgent(
            retriever,
            InferenceEngineBuilder.build_inference_engine(config.get("inference_engine")),
            chat_history = config.get("chat_history"),
            system_prompts_lang = config.get("system_prompts_lang")
        )