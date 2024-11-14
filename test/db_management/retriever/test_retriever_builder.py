import pytest

from vectoria_lib.db_management.retriever.retriever_builder import RetrieverBuilder
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.db_management.retriever.retriever_base import RetrieverBase
from vectoria_lib.db_management.vector_store.vectore_store_builder import VectorStoreBuilder
from langchain_core.documents import Document

def test_build_faiss_retriever(config, data_dir):

    # Empty vector store
    with pytest.raises(ValueError):
        RetrieverBuilder().build(
            config.get("retriever"),
            vector_store = FaissVectorStore(**config.get("vector_store"))
        )


    # Valid vector store
    vector_store = VectorStoreBuilder().build(
        config.get("vector_store"),
        index_path = data_dir / "index" / "BAAI__bge-m3_faiss_index"
    )

    # Valid retriever
    retriever = RetrieverBuilder().build(
        config.get("retriever"),
        vector_store = vector_store
    )
    assert isinstance(retriever, RetrieverBase)
    assert retriever.search_type == config.get("retriever", "search_type")

    docs = retriever.as_langchain_retriever().invoke("What is the capital of France?")
    assert isinstance(docs, list)
    assert len(docs) == config.get("retriever", "top_k")
    assert isinstance(docs[0], Document)
    

