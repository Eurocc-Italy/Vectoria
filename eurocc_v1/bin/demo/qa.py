import argparse
from pathlib import Path
from eurocc_v1.lib.index.faiss_vector_store import FaissVectorStore
from eurocc_v1.lib.rag.faiss_retriever import FaissRetriever
from eurocc_v1.lib.agents.qa_agent import QAAgent

def cli():
    parser = argparse.ArgumentParser(description="EuroCC v1 demo")
    parser.add_argument("-i", "--faiss-index", type=str, required=True, help="")
    return parser.parse_args()

def main(args):

    model_name = Path(args.faiss_index).stem.split("_faiss_index")[0].replace('__', '/')
    faiss_index_bytes = Path(args.faiss_index).read_bytes()
    
    # Load vector store
    vector_store = FaissVectorStore.load_from_pickle(model_name, faiss_index_bytes)

    # Create rag retriever
    retriever = FaissRetriever(search_kwargs={"k": 5})
    retriever.set_retriever(vector_store.as_retriever())

    # Create QA agent
    qa_agent = QAAgent(retriever)

    while True:
        query = input("Query: ")
        if query == "exit":
            break
        docs = qa_agent.ask(query)
        print(docs)


if __name__ == "__main__":
    main(cli())