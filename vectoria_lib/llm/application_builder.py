#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
import logging

from vectoria_lib.applications.qa import QAApplication
from vectoria_lib.common.config import Config
from vectoria_lib.rag.postretrieval_steps.full_paragraphs import FullParagraphs
from vectoria_lib.rag.vector_store.vectore_store_builder import VectorStoreBuilder
from vectoria_lib.rag.retriever.retriever_builder import RetrieverBuilder
from vectoria_lib.llm.llm_builder import LLMFactory
from vectoria_lib.applications.chains import create_qa_chain
from vectoria_lib.llm.prompts.prompt_builder import PromptBuilder
from vectoria_lib.llm.parser import CustomResponseParser
from vectoria_lib.rag.postretrieval_steps.huggingface_reranker import Reranker

class ApplicationBuilder:
    
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
            reranker_llm = LLMFactory.build_llm(config.get("reranker")["inference_engine"])
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
    def build_qa(
        **kwargs: dict
    ) -> QAApplication:
        config = Config()
        retriever_config, reranker_config, full_paragraphs_retriever_config = ApplicationBuilder._create_chain_configuration(kwargs)
        
        if retriever_config is not None:
            ApplicationBuilder.logger.info("Creating QA application with RAG retriever")
            chain = create_qa_chain(
                PromptBuilder(config.get("system_prompts_lang")).get_qa_prompt(),
                LLMFactory.build_llm(config.get("inference_engine")),
                CustomResponseParser(),
                retriever_config = retriever_config,
                reranker_config = reranker_config,
                full_paragraphs_retriever_config = full_paragraphs_retriever_config
            )
            return QAApplication(chain)
        else:
            ApplicationBuilder.logger.info("Creating QA application without the RAG retriever")
            chain_no_retriever = create_qa_chain(
                PromptBuilder(config.get("system_prompts_lang")).get_qa_prompt(),
                LLMFactory.build_llm(config.get("inference_engine")),
                CustomResponseParser(),
                retriever_config = None,
                reranker_config = reranker_config,
                full_paragraphs_retriever_config = full_paragraphs_retriever_config
            )
            return QAApplication(chain_no_retriever)