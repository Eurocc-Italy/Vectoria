import os
import pytest 

from vectoria_lib.db_management.retriever.faiss_retriever import FaissRetriever
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.config import Config
@pytest.mark.parametrize("k",[1,2,3])
def test_faiss_retriever(k):

    vector_store = FaissVectorStore.load_from_pickle(
        TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )
    
    retriever = FaissRetriever(vector_store, search_type="mmr", search_kwargs={"k": k, "fetch_k": k, "lambda_mult": 0.5})

    query = "Who are the actors?"
    docs = retriever.as_langchain_retriever().get_relevant_documents(query)
    assert len(docs) == k

def test_faiss_retriever_full_paragraph(config):
    config.set("system_prompts_lang", "it")

    vector_store = FaissVectorStore.load_from_pickle(
        TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_19_11_24_all_docs.pkl" # TODO: remove this index!
    )
    
    retriever = FaissRetriever(
        vector_store, 
        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 5, "lambda_mult": 0.5})

    query = "Applicabilità?"

    docs = retriever.as_langchain_retriever().invoke(query)

    docs = retriever.retrieve_full_paragraphs(docs)

