import pytest 

from langchain_core.documents import Document

from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.db_management.postretrieval_steps.full_paragraphs import FullParagraphs

def test_full_paragraphs(config, data_dir):
    
    vector_store = FaissVectorStore(
        model_name = config.get("vector_store", "model_name"),
        device = config.get("vector_store", "device"),
        normalize_embeddings = config.get("vector_store", "normalize_embeddings")
    ).load_from_disk(
        data_dir / "index" / "BAAI__bge-m3_faiss_index"
    )
    retriever = FullParagraphs(vector_store)
    
    # No chunks found for the given filter metadata: {'doc_file_name': 'test_file_name', 'paragraph_number': ['1', '2']}
    with pytest.raises(ValueError):
        docs = retriever.post_process([
            Document(page_content="This is a test document", metadata={
                "source": "test_source", 
                "doc_file_name": "test_file_name", 
                "paragraph_number": "1"
            }),
            Document(page_content="This is another test document", metadata={
                "source": "test_source", 
                "doc_file_name": "test_file_name", 
                "paragraph_number": "2"
            })
        ])

