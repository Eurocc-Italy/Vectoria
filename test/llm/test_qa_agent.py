import pytest
import torch

from vectoria_lib.llm.agent_builder import AgentBuilder
from langchain.docstore.document import Document


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent_huggingface(config, index_test_folder, clear_inference_engine_cache):
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
    config.set("inference_engine", value=inference_config)
    config.set("retriever", "top_k", 1)
    _run_engine_test("test_qa_agent_huggingface", index_test_folder)

@pytest.mark.slow
def test_qa_agent_vllm(config, index_test_folder, clear_inference_engine_cache, vllm_server_status_fn):
    inference_config = {
        "name": "vllm",
        "model_name": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "url": "http://localhost:8899/v1",
        "api_key": "abcd"
    }
    if not vllm_server_status_fn(inference_config):
        pytest.skip("VLLM server is not running")

    config.set("inference_engine", value=inference_config)
    config.set("retriever", "top_k", 1)
    _run_engine_test("test_qa_agent_openai", index_test_folder)

def _run_engine_test(run_name, index_test_folder):
    agent = AgentBuilder.build_qa_agent(
        index_path=index_test_folder,
    )
    answer = agent.ask("What is the name of the Vaswani paper?", config={"run_name": run_name})
    assert "answer" in answer
    assert "context" in answer
    assert "input" in answer
    assert len(answer["answer"]) > 0


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent_without_retriever(config, index_test_folder, clear_inference_engine_cache):
    config.set("retriever", "enabled", False)

    agent = AgentBuilder.build_qa_agent(
        index_path=index_test_folder
    )
    context = [
        Document(
            page_content="Deep learning in Natural Language Processing (NLP) is resource-intensive, and the energy and policy considerations are becoming increasingly important as models grow in size and complexity.",
            metadata={"doc_file_name": "test_docx.docx", "paragraph_name": "Background", "paragraph_number": "1"}
        ),
        Document(
            page_content="Energy consumption: Deep learning models, especially large NLP models, require significant computational resources, leading to high energy consumption during both training and inference.",
            metadata={"doc_file_name": "test_docx.docx", "paragraph_name": "Energy consumption", "paragraph_number": "2"}
        ),
        Document(
            page_content="Environmental impact: The energy-intensive nature of deep learning can contribute to increased carbon emissions, raising concerns about sustainability in AI development.",
            metadata={"doc_file_name": "test_docx.docx", "paragraph_name": "Environmental impact", "paragraph_number": "3"}
        ),
        Document(
            page_content="Model efficiency: Optimizing model architectures, using more energy-efficient hardware, and employing techniques like model pruning or quantization can reduce energy demands without sacrificing performance.",
            metadata={"doc_file_name": "test_docx.docx", "paragraph_name": "Model efficiency", "paragraph_number": "4"}
        ),
        Document(
            page_content="Policy frameworks: Governments and organizations are exploring regulations to ensure responsible AI practices, focusing on transparency, sustainability, and the ethical implications of energy use in AI technologies.",
            metadata={"doc_file_name": "test_docx.docx", "paragraph_name": "Policy frameworks", "paragraph_number": "5"}
        )
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
def test_qa_agent_without_retriever_with_reranker(config, index_test_folder, clear_inference_engine_cache):
    config.set("retriever", "enabled", False)
    config.set("reranker", "enabled", True)
    config.set("reranker", "reranked_top_k", 3)

    agent = AgentBuilder.build_qa_agent(
        index_path=index_test_folder
    )
    context = [
        Document(
            page_content="The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU",
            metadata={"doc_file_name": "test_docx.docx", "paragraph_name": "Background", "paragraph_number": "1"}
        ),
        Document(
            page_content="The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as: Attention(Q, K, V ) = softmax(QKT√dk)V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of √1/dk.",
            metadata={"doc_file_name": "test_docx.docx", "paragraph_name": "Scaled Dot-Product Attention", "paragraph_number": "4.2.1"}
        ),
        Document(
            page_content="An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.",
            metadata={"doc_file_name": "test_docx.docx", "paragraph_name": "Attention", "paragraph_number": "4.2"}
        )
    ]
    
    result = agent.ask(
        "Which are the most commonly used attention functions?",
        context=context,
        config={"run_name": "test_qa_agent_without_retriever_with_reranker"}
    )
    assert isinstance(result, dict)
    assert result.keys() == {"input", "context", "docs", "reranked_docs", "answer"}
    assert len(result["reranked_docs"]) == 3
    assert "additive attention" in result["reranked_docs"][0].page_content


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent_without_retriever_with_reranker_with_full_paragraphs(config, index_test_folder, clear_inference_engine_cache):
    config.set("retriever", "enabled", False)
    config.set("reranker", "enabled", True)
    config.set("reranker", "reranked_top_k", 3)
    config.set("full_paragraphs_retriever", "enabled", True)

    agent = AgentBuilder.build_qa_agent(
        index_path=index_test_folder
    )
    context = [
        Document(
            page_content="The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU",
            metadata={"doc_file_name": "test_docx.docx", "paragraph_number": "3", "paragraph_name": "Background"}
        ),
        Document(
            page_content="The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as: Attention(Q, K, V ) = softmax(QKT√dk)V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of √1/dk.",
            metadata={"doc_file_name": "test_docx.docx", "paragraph_number": "4.2.1", "paragraph_name": "Scaled Dot-Product Attention"}
        ),
        Document(
            page_content="An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.",
            metadata={"doc_file_name": "test_docx.docx", "paragraph_number": "4.2", "paragraph_name": "Attention"}
        )
    ]
    
    result = agent.ask(
        "Which are the most commonly used attention functions?",
        context=context,
        config={"run_name": "test_qa_agent_without_retriever_with_reranker_with_full_paragraphs"}
    )
    assert isinstance(result, dict)
    assert result.keys() == {"input", "context", "docs", "reranked_docs", "full_paragraphs_docs", "answer"}
    assert len(result["full_paragraphs_docs"]) == 3
    assert "additive attention" in result["reranked_docs"][0].page_content
    for i, doc in enumerate(result["full_paragraphs_docs"]):
        assert len(doc.page_content) > len(result["reranked_docs"][i].page_content)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent_with_retriever_without_reranker_with_full_paragraphs(config, index_test_folder, clear_inference_engine_cache):
    config.set("retriever", "top_k", 2)
    config.set("full_paragraphs_retriever", "enabled", True)

    agent = AgentBuilder.build_qa_agent(
        index_path=index_test_folder
    )
    
    result = agent.ask(
        "Which are the most commonly used attention functions?",
        config={"run_name": "test_qa_agent_without_retriever_without_reranker_with_full_paragraphs"}
    )
    assert isinstance(result, dict)
    assert result.keys() == {"input", "context", "docs", "full_paragraphs_docs", "answer"}
    assert len(result["full_paragraphs_docs"]) == 2
    assert "additive attention" in result["answer"].lower()




@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent_with_retriever_with_reranker_with_full_paragraphs(config, index_test_folder, clear_inference_engine_cache):
    config.set("reranker", "enabled", True)
    config.set("retriever", "top_k", 5)
    config.set("reranker", "reranked_top_k", 3)
    config.set("full_paragraphs_retriever", "enabled", True)

    agent = AgentBuilder.build_qa_agent(
        index_path=index_test_folder
    )
    
    result = agent.ask(
        "Which are the most commonly used attention functions?",
        config={"run_name": "test_qa_agent_without_retriever_without_reranker_with_full_paragraphs"}
    )
    assert isinstance(result, dict)
    assert result.keys() == {"input", "context", "docs", "reranked_docs", "full_paragraphs_docs", "answer"}
    assert len(result["docs"]) == 5
    assert len(result["reranked_docs"]) == 3
    assert len(result["full_paragraphs_docs"]) == 3
    assert "additive attention" in result["answer"].lower()







@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skip(reason="Chat history needs to be refactored")
def test_qa_agent_with_history(config, index_test_folder, clear_inference_engine_cache):
    config.set("documents_format", "pdf")
    config.set("chat_history", {"enabled": True})

    agent = AgentBuilder.build_qa_agent(
        index_path=index_test_folder
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

