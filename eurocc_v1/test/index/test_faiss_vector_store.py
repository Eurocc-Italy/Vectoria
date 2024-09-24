
from pathlib import Path
from eurocc_v1.lib.index.faiss_vector_store import FaissVectorStore
from eurocc_v1.paths import DATA_DIR
from eurocc_v1.lib.preprocessing.chunking import make_chunks 

def test_save_on_disk_and_load():
    
    text = " ".join(["This is a test sentence"] * 10)
    docs = make_chunks(text)

    vector_store = FaissVectorStore("BAAI/bge-m3").make_index(docs)
    pkl_path, fvs = vector_store.dump_to_pickle(Path(__file__).parent)
    
    assert pkl_path.exists()
    assert isinstance(fvs, FaissVectorStore) 

    vector_store = FaissVectorStore.load_from_pickle(pkl_path)
    assert vector_store.index is not None




