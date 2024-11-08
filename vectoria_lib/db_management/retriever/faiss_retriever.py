#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from typing import List
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
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs)

    def as_langchain_retriever(self):
        """
        Return the FAISS retriever in a format compatible with Langchain retrievers.

        Returns:
        - retriever: The current FAISS retriever instance.
        """
        return self.retriever

    def retrieve_full_paragraphs(self, chunks: List[Document]):
        """
        This method retrieve the full paragraph for a given set of chunks (1 or more depending on config)

        Parameters:
        - chunks: List[Document]

        Returns:
        - str (full paragraph) or List[str] (list of full paragraph)
        """
        # 1) create a new retreiver specialized for full paragraph w/ k and fetch_k set to very high value
        logger.debug("Creating Faiss retriever")
        start_time = time.time()
        self.full_paragraph_retriever = self.vector_store.as_retriever( # TODO: do we want a separate config for this retriever? 
            search_type="mmr", # config.get("FP_retriever_search_type")
            search_kwargs={
                "k": 1000000, # config.get("FP_retriever_top_k")
                "fetch_k": 1000000, #config.get("FP_retriever_fetch_k")
                "lambda_mult": 0.5 # config.get("FP_retriever_lambda_mult")
            }
        )        
        logger.info("Set FULL PARAGRAPHS Faiss retriever in %.2f seconds", time.time() - start_time)
        
        # 2) Extract relevant metadata from already retrieved chunks
        # metadata={
        #   'layout_tag': 'Table',
        #   'doc_file_name': 'PRO250-I-BS-E rev.01 - Istruzioni particolari applicazione modalità Generazione e Approvazione RdA.docx',
        #   'paragraph_number': '3.2',
        #   'paragraph_name': 'Template/Form/Checklist',
        #   'doc_id': 'Codice: PRO250-I-BS-E rev. 01',
        #   'doc_date': 'Data: 23/07/2024',
        #   'doc_type': 'Tipo documento: ISTRUZIONE OPERATIVA',
        #   'seq_id': 0
        # }
        top_1: Document = chunks[0] # TODO: naive version that retrieve full paragraph only for the most relevant chunk
        filter_metadata = {
            "paragraph_number": top_1.metadata["paragraph_number"],
            "doc_id": top_1.metadata["doc_id"]
        }

        # 3) use correct metadata to filter the retrived additional docs
        full_paragraph_chunks: List[Document] = self.retriever.invoke(
            query = "blablabla",
            #filter={"source": "news"}
            filter = filter_metadata
        )

        # 4) Sort retrieved chunks to correct order
        full_paragraph_chunks.sort(key=lambda doc: doc.metadata.get("seq_id", 0))

        # 5) Join page content to re-build full paragraph
        full_paragraph: str = "".join(doc.page_content for i, doc in enumerate(full_paragraph_chunks))

        return full_paragraph
