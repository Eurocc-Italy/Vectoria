import logging
from pathlib import Path
from vectoria_lib.common.utils import Singleton
from vectoria_lib.rag.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.common.config import Config
class VectorStoreBuilder(metaclass=Singleton):

    CACHE = {}

    def __init__(self):
        self.logger = logging.getLogger("db_management")

    def build(self, config: dict, index_path: str | Path = None):
        name = config["name"]
        if name in self.CACHE:
            self.logger.info("Vector store %s already in cache. Returning cached instance." % name)
            vs = self.CACHE[name]
        else:
            self.logger.info("Vector store %s not in cache. Building new instance." % name)
            vs = self._build_vector_store(config)
            self.CACHE[name] = vs

        if index_path:
            self.logger.info("Loading index from disk: %s" % index_path)
            vs.load_from_disk(index_path)

        return vs

    def _build_vector_store(self, config: dict):
        if config["name"] == "faiss":
            return FaissVectorStore(**config)
        else:
            raise ValueError("Invalid vector store name: %s" % config["name"])
