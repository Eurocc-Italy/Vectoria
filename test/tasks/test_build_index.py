import os
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from vectoria_lib.tasks.build_index import build_index
from vectoria_lib.common.paths import TEST_DIR

@pytest.mark.parametrize("extraction_fn", ["extract_text_from_docx_file", "extract_text_from_pdf_file"])
def test_build_index(config, extraction_fn):
    config.set("vector_store", "device", "cuda")

    with TemporaryDirectory() as temp_dir:
        
        doc_format = "docx" if "docx" in extraction_fn else "pdf"
        
        config.set("data_ingestion", "extraction", {
            "format": doc_format,
            "dump_doc_structure_on_file": True,
            "regexes_for_metadata_extraction": []
        })
        
        args = {
            "input_docs_dir" : TEST_DIR / "data" / doc_format,
            "output_index_dir" : Path(temp_dir) / doc_format
            }

        fvs_path, fvs = build_index(**args)

        assert fvs_path.exists()
        assert len(os.listdir(fvs_path)) == 2
        
        assert isinstance(fvs.hf_embedder, HuggingFaceBgeEmbeddings)
        assert isinstance(fvs.index, FAISS)
