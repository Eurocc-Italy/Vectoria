import os
from tempfile import TemporaryDirectory
from vectoria_lib.tasks.build_index import build_index
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.config import Config

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import pytest

@pytest.mark.parametrize("extraction_fn", ["extract_text_from_docx_file", "extract_text_from_pdf_file"])
def test_build_index(config, extraction_fn):
    from pathlib import Path
    config.set("vectoria_logs_dir", value=Path("./test_build_index_logs"))
    with TemporaryDirectory() as temp_dir:

        config.config["pp_steps"][0] = {
            "name": extraction_fn,
            "dump_doc_structure_on_file": True,
            "regexes_for_metadata_extraction": []
        }
        
        if "docx" in extraction_fn:
            doc_format = "docx"
        elif "pdf" in extraction_fn:
            doc_format = "pdf"
        from pathlib import Path
        args = {
            "input_docs_dir" : TEST_DIR / "data" / doc_format,
            "output_dir" : Path(temp_dir) / doc_format
            }

        fvs_path, fvs = build_index(**args)

        assert fvs_path.exists()
        assert len(os.listdir(fvs_path)) == 2
        assert fvs.model_name == "BAAI/bge-m3"
        assert isinstance(fvs.hf_embedder, HuggingFaceBgeEmbeddings)
        assert isinstance(fvs.index, FAISS)
