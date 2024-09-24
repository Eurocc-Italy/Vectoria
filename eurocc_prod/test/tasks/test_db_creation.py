from tempfile import TemporaryDirectory

import argparse
from vectoria_lib.tasks.db_creation import create_and_write_index
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def test_db_creation():

    with TemporaryDirectory() as temp_dir:

        args = argparse.Namespace(**{
            "input_docs_dir" : TEST_DIR / "data/docx",
            "output_index_dir" : temp_dir,
            "hf_embedder_model_name" : "BAAI/bge-m3"
            }
        )

        fvs_path, fvs = create_and_write_index(args)

        assert fvs_path.exists()
        assert fvs_path.is_file()
        assert fvs_path.stat().st_size > 0
        
        assert fvs.model_name == "BAAI/bge-m3"
        assert isinstance(fvs.hf_embedder, HuggingFaceBgeEmbeddings)
        assert isinstance(fvs.index, FAISS)
