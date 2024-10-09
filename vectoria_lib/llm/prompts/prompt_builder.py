from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from vectoria_lib.llm.helpers import get_prompt

class PromptBuilder:

    def __init__(self, lang: str):
        self.lang = lang

    def get_qa_prompt(self):
        return ChatPromptTemplate.from_messages([
                get_prompt("qa.txt", self.lang),
                ("human", "{input}")
        ])


    def get_qa_prompt_with_history(self):        
        return ChatPromptTemplate.from_messages([
                ("system", get_prompt("qa.txt", self.lang)),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
        ])


    def get_contextualize_q_prompt(self):
        return ChatPromptTemplate.from_messages([
                ("system", get_prompt("contextualize_q.txt", self.lang)),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
        ])