#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from typing import List, Set
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import format_document
from langchain.docstore.document import Document
import logging
from vectoria_lib.common.config import Config
import time

logger = logging.getLogger('db_management')
config = Config()

class FaissRetriever:

    def __init__(self, vector_store: FAISS, search_type: str, search_kwargs: dict):
        """
        Constructor that stores the vector_store as it is and a retriever with custom config

        Parameters:
        - vector_store: FAISS
        - search_type: str (e.g. "mmr")
        - search_kwargs: disct (other configs: "k", "fetch_k", "lambda_mult")
        """
        self.retriever = vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs)

    def as_langchain_retriever(self):
        """
        Return the FAISS retriever in a format compatible with Langchain retrievers.

        Returns:
        - retriever: The current FAISS retriever instance.
        """
        return self.retriever
