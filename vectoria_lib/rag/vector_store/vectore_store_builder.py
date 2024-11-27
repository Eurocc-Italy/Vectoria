import logging
from pathlib import Path
from vectoria_lib.rag.vector_store.faiss_vector_store import FaissVectorStore

class VectorStoreBuilder:

    CACHE = {}
    logger = logging.getLogger("db_management")

    @staticmethod
    def build(config: dict, index_path: str | Path = None):
        
        name = config["name"]

        if name in VectorStoreBuilder.CACHE:
            VectorStoreBuilder.logger.info("Vector store %s already in cache. Returning cached instance." % name)
            vs = VectorStoreBuilder.CACHE[name]
        else:
            VectorStoreBuilder.logger.info("Vector store %s not in cache. Building new instance." % name)
            vs = VectorStoreBuilder._build_vector_store(config)
            VectorStoreBuilder.CACHE[name] = vs

        if index_path:
            VectorStoreBuilder.logger.info("Loading index from disk: %s" % index_path)
            vs.load_from_disk(index_path)

        return vs

    @staticmethod
    def _build_vector_store(config: dict):
        if config["name"] == "faiss":
            VectorStoreBuilder.logger.info("Building FAISS vector store")
            return FaissVectorStore(**config)
        else:
            raise ValueError("Invalid vector store name: %s" % config["name"])
