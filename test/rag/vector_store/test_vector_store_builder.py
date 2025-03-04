from vectoria_lib.components.vector_store.vectore_store_builder import VectorStoreBuilder
from vectoria_lib.components.vector_store.faiss_vector_store import FaissVectorStore

def test_build_faiss_vector_store(config, index_test_folder):
    
    vector_store = VectorStoreBuilder.build(
        config.get("vector_store"), 
        index_path=None
    )
    assert isinstance(vector_store, FaissVectorStore)

    vector_store = VectorStoreBuilder.build(
        config.get("vector_store"), 
        index_path = index_test_folder
    )
    assert isinstance(vector_store, FaissVectorStore)
    assert vector_store.index is not None