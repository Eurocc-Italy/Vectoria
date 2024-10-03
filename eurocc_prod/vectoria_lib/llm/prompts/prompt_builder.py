from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from vectoria_lib.llm.helpers import get_prompt

class PromptBuilder:

    @staticmethod
    def get_qa_prompt():
        return ChatPromptTemplate.from_messages([
                get_prompt("qa.txt"),
                ("human", "{input}")
        ])

    @staticmethod
    def get_qa_prompt_with_history():        
        return ChatPromptTemplate.from_messages([
                ("system", get_prompt("qa.txt")),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
        ])

    @staticmethod
    def get_contextualize_q_prompt():
        return ChatPromptTemplate.from_messages([
                ("system", get_prompt("contextualize_q.txt")),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
        ])