from pathlib import Path
from vectoria_lib.db_management.preprocessing.chunking import recursive_character_text_splitter
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from langchain.docstore.document import Document

def test_faiss_vector_store():

    doc = Document(
        page_content = 'The Matrix is a 1999 science fiction action film[5][6] written and directed by the Wachowskis.[a] It is the first installment in the Matrix film series, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving and Joe Pantoliano, and depicts a dystopian future in which humanity is unknowingly trapped inside the Matrix, a simulated reality that intelligent machines have created to distract humans while using their bodies as an energy source.[7] When computer programmer Thomas Anderson, under the hacker alias "Neo", uncovers the truth, he joins a rebellion against the machines along with other people who have been freed from the Matrix.',
        metadata = dict(name="The Matrix", level=0, id=0)
    )
    
    docs = recursive_character_text_splitter(doc, chunk_size=50, chunk_overlap=20)

    model_name = "BAAI/bge-m3"
    vector_store = FaissVectorStore(model_name).make_index(docs)
    pkl_path = vector_store.dump_to_pickle(Path(__file__).parent)
    
    assert pkl_path.exists()

    vector_store = FaissVectorStore.load_from_pickle(pkl_path)
    

    assert vector_store.index is not None