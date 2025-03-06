from vectoria_lib.components.vector_store.vector_store_factory import VectorStoreFactory
from vectoria_lib.components.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.components.vector_store.vectore_store_base import VectorStoreBase

def test_build_faiss_vector_store(config, index_test_folder):
    
    vector_store = VectorStoreFactory.create_vector_store(
        **config.get("vector_store"),
        index_path=None
    )
    assert isinstance(vector_store, FaissVectorStore)

    VectorStoreBase.reset() # destroy the singleton instance
    vector_store = VectorStoreFactory.create_vector_store(
        **config.get("vector_store"),
        index_path = index_test_folder
    )
    assert isinstance(vector_store, FaissVectorStore)
    assert vector_store.index is not None