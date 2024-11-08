#
# VECTORIA
#
# @authors : Andrea Proia, Leonardo Baroncelli
#

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from typing import Sequence
from typing_extensions import Annotated, TypedDict

class StatefulWorkflow:

    @staticmethod
    def to_stateful_workflow(
        chain
        
    ):
        ### Statefully manage chat history ###
        # We define a dict representing the state of the application.
        # This state has the same input and output keys as `chain`.
        class State(TypedDict):
            input: str
            chat_history: Annotated[Sequence[BaseMessage], add_messages]
            context: str
            answer: str

        # We then define a simple node that runs the `chain`.
        # The `return` values of the node update the graph state, so here we just
        # update the chat history with the input message and response.
        def call_model(state: State):
            response = chain.invoke(state)
            return {
                "chat_history": [
                    HumanMessage(state["input"]),
                    AIMessage(response["answer"]),
                ],
                "context": response["context"],
                "answer": response["answer"],
            }

        # Our graph consists only of one node:
        workflow = StateGraph(state_schema=State)
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)

        # Finally, we compile the graph with a checkpointer object.
        # This persists the state, in this case in memory.
        memory = MemorySaver()

        return workflow.compile(checkpointer=memory)
