#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from vectoria_lib.common.paths import ETC_DIR
from vectoria_lib.common.constants import ALLOWED_LANGS
class PromptBuilder:

    def __init__(self, lang: str):
        self.lang = lang

    def get_qa_prompt(self):
        return ChatPromptTemplate.from_messages([
                PromptBuilder.get_prompt("qa.txt", self.lang),
                ("human", "{input}")
        ])


    def get_qa_prompt_with_history(self):        
        return ChatPromptTemplate.from_messages([
                ("system", PromptBuilder.get_prompt("qa.txt", self.lang)),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
        ])


    def get_contextualize_q_prompt(self):
        return ChatPromptTemplate.from_messages([
                ("system", PromptBuilder.get_prompt("contextualize_q.txt", self.lang)),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
        ])
    
    @staticmethod
    def get_prompt(file_name: str, lang):
        if lang not in ALLOWED_LANGS:
            raise ValueError(f"lang '{lang}' is not valid. Supported: 'it', 'eng'.")
        custom_prompt = ETC_DIR / "custom" / "prompts" / lang / file_name
        if custom_prompt.exists():
            return custom_prompt.read_text()
        else:
            return (ETC_DIR / "default" / "prompts" / lang / file_name).read_text()    
        