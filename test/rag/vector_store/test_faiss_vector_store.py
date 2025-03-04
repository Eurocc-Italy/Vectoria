from vectoria_lib.components.preprocessing.chunking import recursive_character_text_splitter
from vectoria_lib.components.vector_store.faiss_vector_store import FaissVectorStore
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores.base import VectorStoreRetriever

def test_faiss_vector_store(config, index_test_folder):

    doc = Document(
        page_content = 'The Matrix is a 1999 science fiction action film[5][6] written and directed by the Wachowskis.[a] It is the first installment in the Matrix film series, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving and Joe Pantoliano, and depicts a dystopian future in which humanity is unknowingly trapped inside the Matrix, a simulated reality that intelligent machines have created to distract humans while using their bodies as an energy source.[7] When computer programmer Thomas Anderson, under the hacker alias "Neo", uncovers the truth, he joins a rebellion against the machines along with other people who have been freed from the Matrix.',
        metadata = dict(source="test", name="The Matrix", level=0, id=0)
    )
    
    docs = recursive_character_text_splitter(doc, chunk_size=50, chunk_overlap=20)
    
    vector_store = FaissVectorStore(
        model_name = config.get("vector_store", "model_name"),
        device = config.get("vector_store", "device"),
        normalize_embeddings = config.get("vector_store", "normalize_embeddings")        
    ).make_index(docs)
    assert isinstance(vector_store.index, FAISS)
    
    pkl_path = vector_store.dump_to_disk("/tmp/test_faiss_vector_store") 

    assert pkl_path.exists()

    vector_store = FaissVectorStore(
        model_name = config.get("vector_store", "model_name"),
        device = config.get("vector_store", "device"),
        normalize_embeddings = config.get("vector_store", "normalize_embeddings")        
    ).load_from_disk(pkl_path)
    assert isinstance(vector_store.index, FAISS)

    docs = vector_store.search("What is the Matrix?", k=1)
    assert len(docs) == 1

    faiss_retriever_config = {
        "search_type": config.get("retriever", "search_type"),
        "search_kwargs": {
            "k": 2,
            "fetch_k": 2,
            "lambda_mult": config.get("retriever", "lambda_mult")
        }
    }
    retriever = vector_store.as_retriever(**faiss_retriever_config)
    assert isinstance(retriever, VectorStoreRetriever)

    docs = retriever.invoke("What is the Matrix?")
    assert len(docs) == 2
