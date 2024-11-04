import logging
import time

from langchain_core.runnables import RunnableLambda
from sentence_transformers import SentenceTransformer, util

from vectoria_lib.db_management.reranking.reranker_base import BaseReranker

class CosSimReranker(BaseReranker):
    # TODO: DEPRECATED
    """
    https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    """
    def __init__(self, args: dict):
        super().__init__(args)

        self.reranked_top_k = self.args["reranked_top_k"]
        self.reranker_path = self.args["model_name"]
        self.logger = logging.getLogger('db_management')

    def rerank(self, inputs: dict) -> list:
        """
        Reranks retrieved documents based on similarity with input query.

        Returns:
        - list: list of top_k reranked chunks
        """
        query: str = inputs["input"]
        docs: list = inputs["docs"]

        # Extract page content and additional metadata
        doc_texts = [doc.page_content for doc in docs]
        doc_metadata = [doc.metadata for doc in docs]

        start_time = time.perf_counter()
        reranker_model = SentenceTransformer(self.reranker_path)
        self.logger.debug("Loading reranker took %.2f seconds", time.perf_counter() - start_time)

        # Convert docs to embeddings using a reranking model (e.g., SBERT)
        doc_embeddings = reranker_model.encode(doc_texts, convert_to_tensor=True)
        query_embedding = reranker_model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity between the query and each document
        scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        
        # Optionally adjust scores with metadata (e.g., document length or recency)
        #adjusted_scores = []
        #for i, score in enumerate(scores):
        #    # Example adjustment based on document length (or other metadata)
        #    length_penalty = 1 / (1 + metadata[i].get('length', 1))  # Simple length penalty
        #    adjusted_scores.append(score.item() * length_penalty)
        adjusted_scores = scores # TODO: implement adjust criteria base on metadata

        # Sort documents by adjusted score (descending)
        sorted_indices = sorted(range(len(adjusted_scores)), key=lambda i: adjusted_scores[i], reverse=True)
        sorted_docs = [docs[i] for i in sorted_indices]

        return sorted_docs[:self.reranked_top_k]
    
    def as_langchain_reranker(self):
        return RunnableLambda(self.rerank)
