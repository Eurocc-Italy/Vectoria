from vectoria_lib.components.vector_store.vector_store_factory import VectorStoreFactory
from vectoria_lib.components.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.components.vector_store.vectore_store_base import VectorStoreBase

def test_build_faiss_vector_store(config, index_test_folder):
    
    vector_store = VectorStoreFactory.build_vector_store(
        **config.get("vector_store")
    )
    assert isinstance(vector_store, FaissVectorStore)

    vector_store.load_index(index_test_folder)

    assert vector_store.index is not None