import logging

from vectoria_lib.common.config import Config
from vectoria_lib.db_management.preprocessing.regex import (
    remove_header,
    remove_footer,
    replace_ligatures,
    remove_bullets,
    remove_empty_lines,
    remove_multiple_spaces  
)
from vectoria_lib.db_management.preprocessing.cleaning import Cleaning
from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline import PreprocessingPipeline
from vectoria_lib.db_management.preprocessing.extraction_pdf import PDFTextExtractor
from vectoria_lib.db_management.preprocessing.extraction_docx import DocXTextExtractor
from vectoria_lib.db_management.preprocessing.chunking import Chunking

logger = logging.getLogger('db_management')

class PreprocessingPipelineBuilder:        
    
    @staticmethod
    def bulding_cleaning_component():
        config = Config()
        cleaning = Cleaning()
        if config.get("remove_header"):
            cleaning.add_cleaning_step(remove_header)
        if config.get("remove_footer"):
            cleaning.add_cleaning_step(remove_footer)
        if config.get("replace_ligatures"):
            cleaning.add_cleaning_step(replace_ligatures)
        if config.get("remove_bullets"):
            cleaning.add_cleaning_step(remove_bullets)
        if config.get("remove_empty_lines"):
            cleaning.add_cleaning_step(remove_empty_lines)
        if config.get("remove_multiple_spaces"):
            cleaning.add_cleaning_step(remove_multiple_spaces)
        return cleaning
    
    @staticmethod
    def building_text_extractor_component():
        config = Config()
        if config.get("documents_format") == "docx":
            return DocXTextExtractor()
        elif config.get("documents_format") == "pdf":
            return PDFTextExtractor()
        else:
            raise ValueError(f"Unsupported document format: {config.get('documents_format')}")
    
    @staticmethod
    def building_chunking_component():
        config = Config()
        return Chunking(config.get("chunk_size"), config.get("chunk_overlap"))

    @staticmethod
    def build_pipeline():
        pipeline = PreprocessingPipeline()
        pipeline.set_text_cleaner(  PreprocessingPipelineBuilder.bulding_cleaning_component())
        pipeline.set_text_extractor(PreprocessingPipelineBuilder.building_text_extractor_component())
        pipeline.set_chunking(      PreprocessingPipelineBuilder.building_chunking_component())
        logger.info("Pipeline built successfully: %s", pipeline)
        return pipeline
