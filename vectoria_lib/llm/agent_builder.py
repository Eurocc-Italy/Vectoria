#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import time
import logging
from pathlib import Path

from vectoria_lib.llm.agents.qa import QAAgent
from vectoria_lib.common.config import Config
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.db_management.retriever.faiss_retriever import FaissRetriever
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
from vectoria_lib.llm.agents.chains import create_qa_chain
from vectoria_lib.llm.prompts.prompt_builder import PromptBuilder
from vectoria_lib.llm.parser import CustomResponseParser


class AgentBuilder:

    @staticmethod
    def _create_retriever_from_faiss_index(
            faiss_index_path: Path,
        ) -> FaissRetriever:
        config = Config()
        logger = logging.getLogger("llm")
        logger.debug("Loading Faiss index: %s", faiss_index_path)

        start_time = time.time()
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
        return retriever
    
    @staticmethod
    def build_qa_agent(
        **kwargs: dict
    ) -> QAAgent:

        config = Config()
        logger = logging.getLogger("llm")
                                        
        retriever_config = None
        if 'faiss_index_path' not in kwargs or kwargs['faiss_index_path'] is None:
            logger.warning("No FAISS index path provided, the RAG retriever will be not be initialized")
        else:
            retriever_config = {
                "retriever": AgentBuilder._create_retriever_from_faiss_index(kwargs['faiss_index_path'])
            }

        reranker_config = None
        if config.get("reranker")["enabled"]:
            reranker_config = {
                "inference_engine": InferenceEngineBuilder.build_inference_engine(config.get("reranker")["inference_engine"]),
                "prompt": None, #PromptBuilder(None).get_reranking_prompt(), TODO: remove this
                "reranked_top_k": config.get("reranker")["reranked_top_k"]
            }

        logger.info("Creating QA agent with the RAG retriever")

        chain_without_retriever = create_qa_chain(
            PromptBuilder(config.get("system_prompts_lang")).get_qa_prompt(),
            InferenceEngineBuilder.build_inference_engine(config.get("inference_engine")),
            CustomResponseParser(),
            retriever_config = None,
            reranker_config = reranker_config
        )

        logger.info("Creating QA agent with the RAG retriever")

        chain_with_retriever = None
        if retriever_config is not None:
            chain_with_retriever = create_qa_chain(
                PromptBuilder(config.get("system_prompts_lang")).get_qa_prompt(),
                InferenceEngineBuilder.build_inference_engine(config.get("inference_engine")),
                CustomResponseParser(),
                retriever_config = retriever_config,
                reranker_config = reranker_config
            )

        # Create QA agent
        return QAAgent(
            chain_without_retriever,
            chain_with_retriever
        )
    

        """
        if self.use_chat_history is True and retriever is not None:

            self.logger.info("Creating QA agent with the RAG retriever and chat history")
            combine_docs_chain = create_generation_chain(
                prompt_builder.get_qa_prompt_with_history(),
                inference_engine,
                output_parser=CustomResponseParser()
            )

            history_aware_retriever = create_history_aware_retriever(
                inference_engine,
                retriever,
                prompt_builder.get_contextualize_q_prompt()
            )

            self.rag_chain = create_retrieval_chain(
                history_aware_retriever,
                combine_docs_chain
            )
            self.rag_chain = StatefulWorkflow.to_stateful_workflow(self.rag_chain)
        """    