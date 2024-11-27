import logging
from vectoria_lib.rag.retriever.retriever_base import RetrieverBase
from vectoria_lib.rag.retriever.faiss_retriever import FaissRetriever
from vectoria_lib.rag.vector_store.vectore_store_base import VectorStoreBase
from vectoria_lib.rag.vector_store.faiss_vector_store import FaissVectorStore


class RetrieverBuilder:

    logger = logging.getLogger("db_management")

    @staticmethod
    def build(
            config: dict,
            vector_store: VectorStoreBase, 
        ) -> RetrieverBase:

        if vector_store is None:
            raise ValueError("A vector store is required to build the retriever")

        if config["name"] == "faiss":
            RetrieverBuilder.logger.info("Building FAISS retriever")
            if not isinstance(vector_store, FaissVectorStore):
                raise ValueError("Vector store must be a FAISS instance")
            
            if vector_store.is_empty():
                raise ValueError("Vector store is empty")
            
            return FaissRetriever(
                search_type = config["search_type"],
                k = config["top_k"],
                fetch_k = config["fetch_k"], 
                lambda_mult = config["lambda_mult"]
            ).from_vector_store(
                vector_store
            )
        
        else:
            raise ValueError(f"Invalid retriever type: {config['name']}")