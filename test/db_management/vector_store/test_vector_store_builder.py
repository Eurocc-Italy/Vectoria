from vectoria_lib.rag.vector_store.vectore_store_builder import VectorStoreBuilder
from vectoria_lib.rag.vector_store.faiss_vector_store import FaissVectorStore

def test_build_faiss_vector_store(config, data_dir):
    vector_store_builder = VectorStoreBuilder()
    vector_store = vector_store_builder.build(
        config.get("vector_store"), 
        index_path=None
    )
    assert isinstance(vector_store, FaissVectorStore)

    vector_store = vector_store_builder.build(
        config.get("vector_store"), 
        index_path=data_dir / "index" / "BAAI__bge-m3_faiss_index"
    )
    assert isinstance(vector_store, FaissVectorStore)
    assert vector_store.index is not None