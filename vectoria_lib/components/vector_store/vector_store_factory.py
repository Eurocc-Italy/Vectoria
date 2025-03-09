from vectoria_lib.components.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.components.vector_store.vectore_store_base import VectorStoreBase

class VectorStoreFactory:
    
    @classmethod
    def create_vector_store(cls, name, **kwargs) -> VectorStoreBase:

        _instances = {}

        if name == "faiss":
            if "faiss" not in _instances:
                _instances["faiss"] = FaissVectorStore(**kwargs)
            return _instances["faiss"]
        
        elif name == "milvus":
            raise NotImplementedError("Milvus vector store is not implemented yet")
        
        else:
            raise ValueError(f"Unknown vector store: {name}")