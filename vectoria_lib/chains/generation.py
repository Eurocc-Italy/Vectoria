from vectoria_lib.common.config import Config
from vectoria_lib.components.llm.llm_factory import LLMFactory
from vectoria_lib.applications.prompt_builder import PromptBuilder
from vectoria_lib.components.llm.parser import CustomResponseParser

def get_generation_chain():
    """
    This function creates a generation chain.
    It takes a system prompt language and builds a prompt from it.
    It then builds a language model from the inference engine and a custom response parser.
    It then returns a generation chain.
    """

    config = Config()
    prompt = PromptBuilder(config.get("system_prompts_lang")).get_qa_prompt()
    llm = LLMFactory.build_llm(config.get("inference_engine")).as_langchain_completion_model()
    output_parser = CustomResponseParser()

    return (

        prompt |

        llm |

        output_parser

    ).with_config(run_name="generation_chain")

