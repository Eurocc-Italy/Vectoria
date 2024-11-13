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
from vectoria_lib.db_management.retriever.full_paragraphs_retriever import FullParagraphsRetriever
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
from vectoria_lib.llm.agents.chains import create_qa_chain
from vectoria_lib.llm.prompts.prompt_builder import PromptBuilder
from vectoria_lib.llm.parser import CustomResponseParser

logger = logging.getLogger("llm")

class AgentBuilder:

    @staticmethod
    def _load_vector_store_from_faiss_index(faiss_index_path: Path) -> FaissVectorStore:
        start_time = time.time()
        logger.debug("Loading Faiss index: %s", faiss_index_path)
        vector_store = FaissVectorStore.load_from_pickle(faiss_index_path)
        logger.info("Loaded Faiss index %s in %.2f seconds", faiss_index_path, time.time() - start_time)
        return vector_store

    @staticmethod
    def _create_retriever_from_faiss_index(vector_store: FaissVectorStore, config: Config) -> FaissRetriever:
        retriever = FaissRetriever(
            vector_store,
            search_type=config.get("retriever", "retriever_search_type"),
            search_kwargs={
                "k": config.get("retriever", "retriever_top_k"),
                "fetch_k": config.get("retriever", "retriever_fetch_k"), 
                "lambda_mult": config.get("retriever", "retriever_lambda_mult")
            }
        )        
        return retriever
    
    @staticmethod
    def _create_full_paragraphs_retriever(config, vector_store: FaissVectorStore) -> FullParagraphsRetriever:
        return FullParagraphsRetriever(vector_store)
    
    @staticmethod
    def _create_chain_configuration(kwargs):
        config = Config()

        if 'faiss_index_path' not in kwargs:
            raise ValueError("No FAISS index path provided")

        vector_store = AgentBuilder._load_vector_store_from_faiss_index(kwargs['faiss_index_path'])
        
        retriever_config = None
        if config.get("retriever", "enabled"):
            retriever_config = {
                "retriever": AgentBuilder._create_retriever_from_faiss_index(vector_store, config)
            }

        reranker_config = None
        if config.get("reranker", "enabled"):
            reranker_config = {
                "inference_engine": InferenceEngineBuilder.build_inference_engine(config.get("reranker")["inference_engine"]),
                "reranked_top_k": config.get("reranker", "reranked_top_k")
            }

            
        logger.info("Creating QA agent with the RAG retriever")
        
        full_paragraphs_retrieval_config = None
        if config.get("full_paragraphs_retrieval", "enable"):
            full_paragraphs_retrieval_config = {
                "retriever": AgentBuilder._create_full_paragraphs_retriever(config, vector_store)
            }
        return retriever_config, reranker_config, full_paragraphs_retrieval_config

    @staticmethod
    def build_qa_agent(
        **kwargs: dict
    ) -> QAAgent:
        config = Config()
        retriever_config, reranker_config, full_paragraphs_retrieval_config = AgentBuilder._create_chain_configuration(kwargs)
        
        if retriever_config is not None:
            chain = create_qa_chain(
                PromptBuilder(config.get("system_prompts_lang")).get_qa_prompt(),
                InferenceEngineBuilder.build_inference_engine(config.get("inference_engine")),
                CustomResponseParser(),
                retriever_config = retriever_config,
                reranker_config = reranker_config,
                full_paragraphs_retrieval_config = full_paragraphs_retrieval_config
            )
            return QAAgent(chain)
        else:
            chain_no_retriever = create_qa_chain(
                PromptBuilder(config.get("system_prompts_lang")).get_qa_prompt(),
                InferenceEngineBuilder.build_inference_engine(config.get("inference_engine")),
                CustomResponseParser(),
                retriever_config = None,
                reranker_config = reranker_config,
                full_paragraphs_retrieval_config = full_paragraphs_retrieval_config
            )
            return QAAgent(chain_no_retriever)



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