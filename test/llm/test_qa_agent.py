import os
import pytest
import torch

from vectoria_lib.llm.agent_builder import AgentBuilder
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.config import Config
from langchain.docstore.document import Document
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder


@pytest.mark.slow
def test_qa_agent_ollama(config, clear_inference_engine_cache, ollama_server_status_fn):
    inference_config = {
        "name": "ollama",
        "model_name": "llama3.2:1b"
    }
    if not ollama_server_status_fn(inference_config):
        pytest.skip("Ollama server is not running")

    config.set("inference_engine", inference_config)
    config.set("retriever", "retriever_top_k", 1)
    _run_engine_test("test_qa_agent_ollama")

@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent_huggingface(config, clear_inference_engine_cache):
    inference_config = {
        "name": "huggingface",
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "device": "cuda",
        "load_in_4bit": True,
        "load_in_8bit": False,
        "max_new_tokens": 100,
        "trust_remote_code": True,
        "device_map": "auto",
        "temperature": 0.1
    }
    config.set("inference_engine", inference_config)
    config.set("retriever", "retriever_top_k", 1)
    _run_engine_test("test_qa_agent_huggingface")

@pytest.mark.slow
def test_qa_agent_vllm(config, clear_inference_engine_cache, vllm_server_status_fn):
    inference_config = {
        "name": "vllm",
        "model_name": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "url": "http://localhost:8899/v1",
        "api_key": "abcd"
    }
    if not vllm_server_status_fn(inference_config):
        pytest.skip("VLLM server is not running")

    config.set("inference_engine", value=inference_config)
    config.set("retriever", "retriever_top_k", 1)
    _run_engine_test("test_qa_agent_openai")

def _run_engine_test(run_name):
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl",
    )
    answer = agent.ask("What is the name of the movie?", config={"run_name": run_name})
    assert "answer" in answer
    assert "context" in answer
    assert "input" in answer
    assert len(answer["answer"]) > 0


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent_without_retriever(config, clear_inference_engine_cache):
    config.set("retriever", "enabled", False)

    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )
    context = [
        Document("Deep learning in Natural Language Processing (NLP) is resource-intensive, and the energy and policy considerations are becoming increasingly important as models grow in size and complexity."),
        Document("Energy consumption: Deep learning models, especially large NLP models, require significant computational resources, leading to high energy consumption during both training and inference."),
        Document("Environmental impact: The energy-intensive nature of deep learning can contribute to increased carbon emissions, raising concerns about sustainability in AI development."),
        Document("Model efficiency: Optimizing model architectures, using more energy-efficient hardware, and employing techniques like model pruning or quantization can reduce energy demands without sacrificing performance."),
        Document("Policy frameworks: Governments and organizations are exploring regulations to ensure responsible AI practices, focusing on transparency, sustainability, and the ethical implications of energy use in AI technologies.")
    ]
    
    result = agent.ask(
        "Which are the energy and policy considerations for deep learning in NLP?",
        context=context,
        config={"run_name": "test_qa_agent_without_retriever"}
    )
    assert isinstance(result, dict)
    assert result.keys() == {"input", "context", "docs", "answer"}
    assert len(result["docs"]) == 5

@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent_without_retriever_with_reranker(config, clear_inference_engine_cache):
    config.set("retriever", "enabled", False)
    config.set("reranker", "enabled", True)
    config.set("reranker", "reranked_top_k", 3)

    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )
    context = [
        Document("Deep learning in Natural Language Processing (NLP) is resource-intensive, and the energy and policy considerations are becoming increasingly important as models grow in size and complexity."),
        Document("Energy consumption: Deep learning models, especially large NLP models, require significant computational resources, leading to high energy consumption during both training and inference."),
        Document("Environmental impact: The energy-intensive nature of deep learning can contribute to increased carbon emissions, raising concerns about sustainability in AI development."),
        Document("Model efficiency: Optimizing model architectures, using more energy-efficient hardware, and employing techniques like model pruning or quantization can reduce energy demands without sacrificing performance."),
        Document("Policy frameworks: Governments and organizations are exploring regulations to ensure responsible AI practices, focusing on transparency, sustainability, and the ethical implications of energy use in AI technologies.")
    ]
    
    result = agent.ask(
        "Which are the energy and policy considerations for deep learning in NLP?",
        context=context,
        config={"run_name": "test_qa_agent_without_retriever_with_reranker"}
    )
    assert isinstance(result, dict)
    assert result.keys() == {"input", "context", "docs", "reranked_docs", "answer"}
    assert len(result["reranked_docs"]) == 3


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
#@pytest.mark.skip(reason="The index used for this test needs to be replaced")
def test_qa_agent_without_retriever_with_reranker_with_full_paragraphs(config, clear_inference_engine_cache):
    config.set("retriever", "enabled", False)
    config.set("reranker", "enabled", True)
    config.set("reranker", "reranked_top_k", 3)
    config.set("full_paragraphs_retrieval", "enable", True)

    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_19_11_24_all_docs.pkl"
    )
    context = [
        Document(page_content='Scopo del presente documento è quello di definire le modalità e i requisiti per la gestione da parte del Procurement di tutte le tipologie di Richieste di Acquisto (RdA) finalizzate all\'emissione del relativo Ordine di Acquisto (OdA). La Gestione Richiesta di Acquisto si attiva con la manifestazione di un fabbisogno, da parte di un ente e/o funzione aziendale, che si identifica nella successiva emissione di una RdA su sistema informativo SAP.', metadata={'layout_tag': 'Paragraph', 'doc_file_name': 'PRO011-P-IT-E rev.01 Gestione delle Richieste di Acquisto.docx', 'paragraph_number': '1.1', 'paragraph_name': 'Scopo', 'doc_id': 'Codice: PRO011-P-IT-E rev. 01.00', 'doc_date': 'Data: 10/05/2024', 'doc_type': 'Tipo documento: PROCEDURA', 'seq_id': 0}),
        Document(page_content='Il presente documento si applica alla Divisione Elettronica di Leonardo S.p.a. (di seguito denominata Divisione), con esclusione di UK Business. La descrizione puntuale delle modalità operative di attuazione dell\'attività di "Gestione delle Richieste d\'Acquisto" è declinata in documenti dedicati.', metadata={'layout_tag': 'Paragraph', 'doc_file_name': 'PRO011-P-IT-E rev.01 Gestione delle Richieste di Acquisto.docx', 'paragraph_number': '1.2', 'paragraph_name': 'Applicabilità', 'doc_id': 'Codice: PRO011-P-IT-E rev. 01.00', 'doc_date': 'Data: 10/05/2024', 'doc_type': 'Tipo documento: PROCEDURA', 'seq_id': 0}),
        Document(page_content='Scopo del presente documento è definire le modalità operative, i ruoli e le responsabilità nell\'emissione degli Ordini d\'Acquisto nell\'ambito, più generale, del processo di approvvigionamento.', metadata={'layout_tag': 'Paragraph', 'doc_file_name': "PRO012-P-IT-I rev.00 - Emissione degli Ordini d'Acquisto_Contratti e degli Accordi Quadro.docx", 'paragraph_number': '1.1', 'paragraph_name': 'Scopo', 'doc_id': 'Codice: PRO012-P-IT-I rev.00', 'doc_date': 'Data: 02/09/2022', 'doc_type': 'Tipo documento: PROCESSO', 'seq_id': 0})
    ]
    
    result = agent.ask(
        "Cosa si intende per Ordine d'Acquisto?",
        context=context,
        config={"run_name": "test_qa_agent_without_retriever_with_reranker_with_full_paragraphs"}
    )
    assert isinstance(result, dict)
    assert result.keys() == {"input", "context", "docs", "reranked_docs", "full_paragraphs_docs", "answer"}
    assert len(result["full_paragraphs_docs"]) == 3


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skip(reason="The index used for this test needs to be replaced")
def test_qa_agent_with_retriever_without_reranker_with_full_paragraphs(config, clear_inference_engine_cache):
    config.set("retriever", "retriever_top_k", 2)
    config.set("full_paragraphs_retrieval", "enable", True)

    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_19_11_24_all_docs.pkl"
    )
    
    result = agent.ask(
        "Cosa si intende per Ordine d'Acquisto?",
        config={"run_name": "test_qa_agent_without_retriever_without_reranker_with_full_paragraphs"}
    )
    assert isinstance(result, dict)
    assert result.keys() == {"input", "context", "docs", "full_paragraphs_docs", "answer"}
    assert len(result["full_paragraphs_docs"]) == 2




@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
#@pytest.mark.skip(reason="The index used for this test needs to be replaced")
def test_qa_agent_with_retriever_with_reranker_with_full_paragraphs(config, clear_inference_engine_cache):
    config.set("reranker", "enabled", True)
    config.set("reranker", "reranked_top_k", 3)
    config.set("full_paragraphs_retrieval", "enable", True)

    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_19_11_24_all_docs.pkl"
    )
    
    result = agent.ask(
        "Cosa si intende per Ordine d'Acquisto?",
        config={"run_name": "test_qa_agent_without_retriever_without_reranker_with_full_paragraphs"}
    )
    assert isinstance(result, dict)
    assert result.keys() == {"input", "context", "docs", "reranked_docs", "full_paragraphs_docs", "answer"}
    assert len(result["full_paragraphs_docs"]) == 3







@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skip(reason="Chat history needs to be refactored")
def test_qa_agent_with_history(config, clear_inference_engine_cache):
    config.set("documents_format", "pdf")
    config.set("chat_history", {"enabled": True})
    config.set("retriever_top_k", 1)

    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_airxv_papers.pkl"
    )

    answer = agent.ask(
        "Which are the energy and policy considerations for deep learning in NLP?",
        session_id="test_session"
    )
    chat_history = agent.get_chat_history("test_session", pretty_print=False)
    assert len(chat_history) == 2

    answer = agent.ask(
        "How these two are related?",
        session_id="test_session"
    )
    chat_history = agent.get_chat_history("test_session", pretty_print=False)
    assert len(chat_history) == 4

    answer = agent.ask(
        "What did I ask before?",
        session_id="test_session_2"
    )
    chat_history = agent.get_chat_history("test_session_2", pretty_print=False)
    assert len(chat_history) == 2

