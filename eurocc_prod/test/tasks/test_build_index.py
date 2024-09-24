from tempfile import TemporaryDirectory

from vectoria_lib.tasks.build_index import build_index
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.config import Config

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import pytest

@pytest.mark.parametrize("doc_format", ["docx", "pdf"])
def test_build_index(doc_format):

    with TemporaryDirectory() as temp_dir:
        Config().set("documents_format", doc_format)
        args = {
            "input_docs_dir" : TEST_DIR / "data" / doc_format,
            "output_index_dir" : temp_dir
            }
        
        fvs_path, fvs = build_index(**args)

        assert fvs_path.exists()
        assert fvs_path.is_file()
        assert fvs_path.stat().st_size > 0
        
        assert fvs.model_name == "BAAI/bge-m3"
        assert isinstance(fvs.hf_embedder, HuggingFaceBgeEmbeddings)
        assert isinstance(fvs.index, FAISS)
