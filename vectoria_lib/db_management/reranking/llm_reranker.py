import logging
import time

from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer, util

from vectoria_lib.db_management.reranking.reranker_base import BaseReranker
from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase

class LLMReranker(BaseReranker):
    # TODO: DEPRECATED

    def __init__(self, args: dict):
        super().__init__(args)
        self.logger = logging.getLogger('db_management')

    def rerank(self, inputs: dict) -> list:
        """
        Reranks retrieved documents based on similarity with input query.

        Returns:
        - dict: input dictionary with reranked_docs_indices key
        """
        reranked_docs_indices = self.inference_engine.invoke(inputs)
        assert isinstance(reranked_docs_indices, list)
        assert len(reranked_docs_indices) <= len(inputs["docs"])

        inputs["reranked_docs"] = [inputs["docs"][i] for i in reranked_docs_indices]

        return inputs
    
    def as_langchain_reranker(self):
        return RunnableLambda(self.rerank)
