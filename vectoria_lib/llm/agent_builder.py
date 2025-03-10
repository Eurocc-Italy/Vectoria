#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
import logging

from vectoria_lib.llm.agents.qa import QAAgent
from vectoria_lib.common.config import Config
from vectoria_lib.rag.postretrieval_steps.full_paragraphs import FullParagraphs
from vectoria_lib.rag.vector_store.vectore_store_builder import VectorStoreBuilder
from vectoria_lib.rag.retriever.retriever_builder import RetrieverBuilder
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
from vectoria_lib.llm.agents.chains import create_qa_chain
from vectoria_lib.llm.prompts.prompt_builder import PromptBuilder
from vectoria_lib.llm.parser import CustomResponseParser
from vectoria_lib.rag.postretrieval_steps.huggingface_reranker import Reranker

class AgentBuilder:
    
    logger = logging.getLogger("llm")
    
    @staticmethod
    def _create_chain_configuration(kwargs):
        config = Config()

        if 'index_path' not in kwargs:
            raise ValueError("No index path provided")

        vector_store = VectorStoreBuilder().build(
            config.get("vector_store"),
            index_path = kwargs['index_path']
        )        

        retriever_config = None
        if config.get("retriever", "enabled"):
            retriever_config = {
                "retriever": RetrieverBuilder().build(config.get("retriever"), vector_store)
            }

        reranker_config = None
        if config.get("reranker", "enabled"):
            reranker_llm = InferenceEngineBuilder.build_inference_engine(config.get("reranker")["inference_engine"])
            reranker_config = {
                "reranker": Reranker(reranker_llm),
                "reranked_top_k": config.get("reranker", "reranked_top_k")
            }

        full_paragraphs_retriever_config = None
        if config.get("full_paragraphs_retriever", "enabled"):
            full_paragraphs_retriever_config = {
                "retriever": FullParagraphs(vector_store)
            }
        return retriever_config, reranker_config, full_paragraphs_retriever_config

    @staticmethod
    def build_qa_agent(
        **kwargs: dict
    ) -> QAAgent:
        config = Config()
        retriever_config, reranker_config, full_paragraphs_retriever_config = AgentBuilder._create_chain_configuration(kwargs)
        
        if retriever_config is not None:
            AgentBuilder.logger.info("Creating QA agent with the RAG retriever")
            chain = create_qa_chain(
                PromptBuilder(config.get("system_prompts_lang")).get_qa_prompt(),
                InferenceEngineBuilder.build_inference_engine(config.get("inference_engine")),
                CustomResponseParser(),
                retriever_config = retriever_config,
                reranker_config = reranker_config,
                full_paragraphs_retriever_config = full_paragraphs_retriever_config
            )
            return QAAgent(chain)
        else:
            AgentBuilder.logger.info("Creating QA agent without the RAG retriever")
            chain_no_retriever = create_qa_chain(
                PromptBuilder(config.get("system_prompts_lang")).get_qa_prompt(),
                InferenceEngineBuilder.build_inference_engine(config.get("inference_engine")),
                CustomResponseParser(),
                retriever_config = None,
                reranker_config = reranker_config,
                full_paragraphs_retriever_config = full_paragraphs_retriever_config
            )
            return QAAgent(chain_no_retriever)