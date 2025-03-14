from vectoria_lib.components.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.components.vector_store.vectore_store_base import VectorStoreBase
from vectoria_lib.components.llm.llm_factory import LLMFactory
class VectorStoreFactory:
    
    @classmethod
    def build_vector_store(cls, **kwargs) -> VectorStoreBase:

        _instances = {}

        if kwargs["name"] == "faiss":
            if "faiss" not in _instances:
                embedder = LLMFactory.build_llm(kwargs["inference_engine"]).as_langchain_embeddings_model()
                _instances["faiss"] = FaissVectorStore(embedder_model=embedder)

            return _instances["faiss"]
        
        elif kwargs["name"] == "milvus":
            raise NotImplementedError("Milvus vector store is not implemented yet")
        
        else:
            raise ValueError(f"Unknown vector store: {kwargs['name']}")