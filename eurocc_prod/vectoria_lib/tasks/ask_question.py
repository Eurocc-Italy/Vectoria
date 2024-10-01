#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#


# ----------------------------------------------------------------------------------------------
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.db_management.retriever.faiss_retriever import FaissRetriever
from vectoria_lib.llm.agent_builder import AgentBuilder

def ask_question(
    **kwargs: dict
):
    qa_agent = AgentBuilder.build_qa_agent(**kwargs)
    answer = qa_agent.ask(kwargs["query"])
    return answer