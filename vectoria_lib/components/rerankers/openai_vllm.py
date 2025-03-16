from typing import Any, Dict, List, Tuple
import operator

import requests
from langchain_core.documents import BaseDocumentCompressor, Document
from typing import Sequence, Optional
from langchain_core.callbacks import Callbacks
from pydantic import Field
from pydantic import ConfigDict

class OpenAIvLLMReranker(BaseDocumentCompressor):
    # Adapted from langchain/libs/langchain/langchain/retrievers/document_compressors/cross_encoder_rerank.py
    openai_api_base: str
    openai_api_key: str
    top_n: int = 5
    model_name: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        scores = self.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        return [doc for doc, _ in result[: self.top_n]]


    def post_http_request(self, prompt: dict) -> requests.Response:
        response = requests.post(self.openai_api_base + "/score", headers={"User-Agent": "Test Client"}, json=prompt, timeout=5)
        return response

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute similarity scores using the vLLM score endpoint.

        Args:
            text_pairs: The list of text text_pairs to score the similarity.

        Returns:
            List of scores, one for each pair.
        """
        text_1 = [t[0] for t in text_pairs]
        text_2 = [t[1] for t in text_pairs]
        prompt = {"model": self.model_name, "text_1": text_1, "text_2": text_2}
        response = self.post_http_request(prompt=prompt).json()
        data = response["data"]
        _ = response["usage"]
        return [d["score"] for d in data if d["object"] == "score"]
    
