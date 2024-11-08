#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import re
import logging
from langchain.docstore.document import Document
logger = logging.getLogger('db_management')

# TODO: optimize regex compilation

def remove_header(doc: Document, regex: str) -> str:
    logging.debug("Removing header")
    doc.page_content = re.compile(regex, re.IGNORECASE).sub("", doc.page_content).strip()
    return doc 
    
def remove_footer(doc: Document, regex: str):
    logging.debug("Removing footer")
    doc.page_content = re.compile(regex, re.IGNORECASE).sub("", doc.page_content).strip()
    return doc

# def remove_empty_lines(doc: Document, regex: str = None) -> str: # TODO: add regex instead of building a temp list    
#     # TODO: optimize me!
#     logging.debug("Removing empty lines")
#     lines = text.splitlines()
#     non_empty_lines = [line for line in lines if line.strip() != '']
#     return '\n'.join(non_empty_lines)

def remove_multiple_spaces(doc: Document, regex: str = None) -> str:
    logging.debug("Removing multiple spaces")
    doc.page_content = re.compile(r"[ \t]{2,}").sub(" ", doc.page_content).strip()
    return doc
    
def replace_ligatures(doc: Document, regex: str = None) -> str:
    logging.debug("Replacing ligatures")
    ligatures = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
        "Ꜳ": "AA",
        "Æ": "AE",
        "ꜳ": "aa",
    }
    # TODO: maybe this could be optimized 
    for search, replace in ligatures.items():
        doc.page_content = doc.page_content.replace(search, replace)        
    return doc

def remove_bullets(doc: Document, regex: str = None) -> str:
    logging.debug("Removing bullets")
    """
    • (\u2022)
    ▪ (\u25AA)
    ➢ (\u27A2)  
    """
    doc.page_content = re.compile(r"^\s*[\u2022\u25AA\u27A2]\s*", flags=re.MULTILINE).sub("", doc.page_content).strip() # Important to add multiline flag
    return doc