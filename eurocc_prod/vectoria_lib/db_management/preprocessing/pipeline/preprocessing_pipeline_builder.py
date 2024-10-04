import logging
from vectoria_lib.common.config import Config
from vectoria_lib.db_management.preprocessing.extraction_pdf import extract_text_from_pdf_file
from vectoria_lib.db_management.preprocessing.extraction_docx import extract_text_from_docx_file
from vectoria_lib.db_management.preprocessing.cleaning import (
    remove_header,
    remove_footer,
    replace_ligatures,
    remove_bullets,
    remove_multiple_spaces  
)
from functools import partial
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from vectoria_lib.db_management.preprocessing.document_data import get_structured_data
from vectoria_lib.db_management.preprocessing.chunking import recursive_character_text_splitter
from vectoria_lib.io.folder_reader import get_files_in_folder
from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline import PreprocessingPipeline

logger = logging.getLogger('db_management')


from langchain.docstore.document import Document

def multiprocessing_wrapper(fn, docs: list[Document]):
    import multiprocessing as mp
    with mp.Pool(processes=mp.cpu_count()) as pool:
        preprocessed_docs = pool.map(fn, docs)
    return preprocessed_docs

class PreprocessingPipelineBuilder:        
        
    @staticmethod
    def get_text_extractor_fn(config):
        if config["format"] == "docx":
            return RunnableLambda(extract_text_from_file_docx)
        elif config["format"] == "pdf":
            return RunnableLambda(extract_text_from_file_pdf)
        else:
            raise ValueError(f"Unsupported document format: {config.get('documents_format')}")
    
    @staticmethod
    def get_cleaning_step_fn(cleaning_step_name: str):
        if cleaning_step_name == "remove_header":
            return RunnableLambda(remove_header)
        elif cleaning_step_name == "remove_footer":
            return RunnableLambda(remove_footer)
        elif cleaning_step_name == "replace_ligatures":
            return RunnableLambda(replace_ligatures)
        elif cleaning_step_name == "remove_bullets":
            return RunnableLambda(partial(multiprocessing_wrapper, remove_bullets))
        elif cleaning_step_name == "remove_multiple_spaces":
            return RunnableLambda(remove_multiple_spaces)
        else:
            raise ValueError(f"Unsupported cleaning step: {cleaning_step_name}")

    @staticmethod
    def get_chunking_fn(chunking_step_name: str):
        if chunking_step_name == "recursive_character_text_splitter":
            return RunnableLambda(recursive_character_text_splitter)
        else:
            raise ValueError(f"Unsupported chunking step: {chunking_step_name}")

    @staticmethod
    def check_fields(pp_config: dict):
        if pp_config[0]["name"] != "extraction":
            raise ValueError("Missing 'extraction' field in preprocessing pipeline configuration")

    

    @staticmethod
    def build_pipeline() -> PreprocessingPipeline:
        pp_steps_config = Config().get("pp_steps")
        # PreprocessingPipelineBuilder.check_fields(pp_steps_config)

        extraction_step = pp_steps_config[0]
        fn_name = extraction_step.pop("name")
        chain = RunnableLambda(globals()[fn_name]).bind(**extraction_step)

        for step in pp_steps_config[1:]:
            fn = globals()[step.pop("name")]
            chain = chain | RunnableLambda(fn).bind(**step).map()

            

        #clean_step = PreprocessingPipelineBuilder.get_cleaning_step_fn("remove_bullets")
        # TODO: how to pass text instead 
        # TOOD: multiprocessing?


        #chain = text_extractor | RunnableLambda(get_structured_data)  
        #chain = chain | clean_step 

        chain.get_graph().print_ascii()



        return PreprocessingPipeline(chain)