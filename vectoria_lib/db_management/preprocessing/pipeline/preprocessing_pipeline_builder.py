import logging
from vectoria_lib.common.config import Config

# ----------------------------------------------------------------------------------------------------------------------
# DO NOT REMOVE THESE IMPORTS, they are get from global namespace
from vectoria_lib.db_management.preprocessing.extraction_pdf import extract_text_from_pdf_file
from vectoria_lib.db_management.preprocessing.extraction_docx import extract_text_from_docx_file
from vectoria_lib.db_management.preprocessing.cleaning import (
    remove_header,
    remove_footer,
    replace_ligatures,
    remove_bullets,
    remove_multiple_spaces  
)
from vectoria_lib.db_management.preprocessing.chunking import recursive_character_text_splitter
# ----------------------------------------------------------------------------------------------------------------------from functools import partial

from langchain_core.runnables import RunnableLambda
from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline_executor import PreprocessingPipelineExecutor

logger = logging.getLogger('db_management')


from langchain.docstore.document import Document

# def multiprocessing_wrapper(fn, docs: list[Document]):
#     import multiprocessing as mp
#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         preprocessed_docs = pool.map(fn, docs)
#     return preprocessed_docs

class PreprocessingPipelineBuilder:

    @staticmethod
    def check_fields(pp_config: dict) -> None:
        """
        Check if the preprocessing pipeline configuration is valid.
        """
        if pp_config[0]["name"] not in ["extract_text_from_docx_file", "extract_text_from_pdf_file"]:
            raise ValueError("First step in preprocessing pipeline must be 'extract_text_from_docx_file' or 'extract_text_from_pdf_file'") 

    @staticmethod
    def build_pipeline() -> PreprocessingPipelineExecutor:
        pp_steps_config = Config().get("pp_steps")
        PreprocessingPipelineBuilder.check_fields(pp_steps_config)

        extraction_step = pp_steps_config[0]
        fn_name = extraction_step.pop("name")
        chain = RunnableLambda(globals()[fn_name]).bind(**extraction_step)

        for step_args in pp_steps_config[1:]:
            fn = globals()[step_args.pop("name")] # get the function from the global namespace
            chain = chain | RunnableLambda(fn).bind(**step_args).map() # with .map() the RunnableLambda is applied to each output of the previous chain

        try:
            chain.get_graph().print_ascii() # print the graph
        except Exception as e:
            logger.warning("Failed to print graph: %s", e)

        return PreprocessingPipelineExecutor(chain)