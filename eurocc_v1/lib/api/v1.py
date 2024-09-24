from pathlib import Path
from eurocc_v1.lib.preprocessing.cleaning import clean_text
from eurocc_v1.lib.preprocessing.chunking import make_chunks
from eurocc_v1.lib.preprocessing.pdf_extraction import extract_text_from_pdf
from eurocc_v1.lib.index.faiss_vector_store import FaissVectorStore
from eurocc_v1.lib.rag.faiss_retriever import FaissRetriever
from eurocc_v1.lib.agents.qa_agent import QAAgent

def create_and_write_index(
    input_docs_dir: str,
    output_index_dir: str,
    hf_embedder_model_name: str
) -> tuple[Path, FaissVectorStore]:
    
    # Preprocessing
    text = extract_text_from_pdf(input_docs_dir)
    text = clean_text(text)
    docs = make_chunks(text)
    
    # Create vector store
    fvs = FaissVectorStore(hf_embedder_model_name).make_index(docs)
    return fvs.dump_to_pickle(output_index_dir), fvs


def create_qa_agent(
    faiss_index_name: str,
    faiss_index: bytes,
    k: int
) -> QAAgent:

    # Load vector store
    vector_store = FaissVectorStore.load_from_pickle(faiss_index_name, faiss_index)

    # Create rag retriever
    retriever = FaissRetriever(search_kwargs={"k": k})
    retriever.set_retriever(vector_store.as_retriever())

    # Create QA agent
    return QAAgent(retriever)