import re
from langchain.schema import BaseOutputParser

class CustomResponseParser(BaseOutputParser):
    
    def filter_prefix(self, text: str):
        match = re.search(r'^(.+\s*=*\s*RISPOSTA\s\s*=*\s)(.+)', text, re.DOTALL)
        if match:
            response = match.group(2)
            response = re.sub(r'\s{2,}', ' ', response).strip()
            return response
        return None
    
    def filter_postfix(self, text: str):
        match = re.search(r'(.+)(\s*Fine Risposta|Fine|Human:)', text)
        if match:
            response = match.group(1)
            response = re.sub(r'\s{2,}', ' ', response).strip()
            return response
        return None

    def parse(self, text: str) -> str:
        print(f"CustomResponseParser: Parsing text: {text}\n\n")
        response = self.filter_prefix(text)
        if not response:
            return f"No valid response found for text:\n {text}"
        response = self.filter_postfix(response)
        if not response:
            return f"\n {text}"
        return response
