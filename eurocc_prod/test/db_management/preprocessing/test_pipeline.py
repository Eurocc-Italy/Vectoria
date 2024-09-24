import pytest

from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline import PreprocessingPipeline
from vectoria_lib.db_management.preprocessing.extraction_docx import DocXTextExtractor
from vectoria_lib.db_management.preprocessing.cleaning import Cleaning
from vectoria_lib.db_management.preprocessing.chunking import Chunking
from vectoria_lib.db_management.preprocessing.regex import remove_empty_lines

def test_pipeline():
    config = Config() 
    config.set("chunk_size", 100)
    config.set("chunk_overlap", 10)
    config.set("documents_format", "docx")

    pp = PreprocessingPipeline()
    pp.set_text_extractor(DocXTextExtractor())
    pp.set_text_cleaner(Cleaning().add_cleaning_step(remove_empty_lines))
    pp.set_chunking(Chunking(config.get("chunk_size"), config.get("chunk_overlap")))
    
    processed_docs = pp.run(TEST_DIR / "data/docx")

    # A list of LangChain Document (chunks) for each input document
    assert len(processed_docs) == 12
