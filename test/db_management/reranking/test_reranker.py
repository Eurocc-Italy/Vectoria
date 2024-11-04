import pytest 

from vectoria_lib.common.paths import TEST_DIR
from langchain.docstore.document import Document

from vectoria_lib.common.config import Config
from vectoria_lib.db_management.reranking.cos_sim_reranker import CosSimReranker
from vectoria_lib.db_management.reranking.llm_reranker import LLMReranker

def test_cos_sim_reranker():
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    config.set("reranker", {
        "model_name": "all-MiniLM-L12-v2",
        "reranked_top_k": 1,
        "name": "cos_sim"
    })
    reranker = CosSimReranker(config.get("reranker"))

    query = "Who are the actors of The Matrix movie of 1999?"

    docs = [
        Document(
            page_content="The Matrix Reloaded is a 2003 science fiction action film written and directed by the Wachowskis."
        ),
        Document(
            page_content="The Matrix is a 1999 science fiction action film written and directed by the Wachowskis, starring Keanu Reeves, Laurence Fishburne, and Carrie-Anne Moss."
        ),
        Document(
            page_content="The Matrix Revolutions is a 2003 science fiction action film written and directed by the Wachowskis."
        )
    ]
    
    reranked_docs = reranker.rerank({"input": query, "docs": docs})
    
    assert len(reranked_docs) == 1


def test_llm_reranker():
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    config.set("reranker", {
        "model_name": "all-MiniLM-L12-v2",
        "reranked_top_k": 1,
        "name": "llm",
        "inference_engine": {
            "name": "huggingface",
            "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "device": "cuda",
            "load_in_8bit": True,
            "max_new_tokens": 150,
            "trust_remote_code": False,
            "device_map": None,
            "temperature": 0.1
        }
    })

    reranker = LLMReranker(config.get("reranker"))

    query = "Who are the actors of The Matrix movie of 1999?"

    docs = [
        Document(
            page_content="The Matrix Reloaded is a 2003 science fiction action film written and directed by the Wachowskis."
        ),
        Document(
            page_content="The Matrix is a 1999 science fiction action film written and directed by the Wachowskis, starring Keanu Reeves, Laurence Fishburne, and Carrie-Anne Moss."
        ),
        Document(
            page_content="The Matrix Revolutions is a 2003 science fiction action film written and directed by the Wachowskis."
        )
    ]
    
    reranked_docs = reranker.rerank({"input": query, "docs": docs})

    assert len(reranked_docs) == 1