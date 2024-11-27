#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import re
import logging
from langchain.docstore.document import Document
logger = logging.getLogger('db_management')

regex_cache = {}

def seek_and_replace(doc: Document, regex_configs: list[dict] = []) -> str:
    for regex_config in regex_configs:
        logger.debug("Seeking and replacing %s with with %s regex %s", regex_config["name"], regex_config["replace_with"], regex_config["pattern"])
        if regex_config["pattern"] not in regex_cache:
            regex_cache[regex_config["pattern"]] = re.compile(regex_config["pattern"], re.IGNORECASE)
        doc.page_content = regex_cache[regex_config["pattern"]].sub(regex_config["replace_with"], doc.page_content)
    return doc

def extract_metadata_from_text(text: str, regex_configs: list[dict] = []) -> dict:
    metadata = {}
    for regex_config in regex_configs:
        logger.debug("Extracting %s from text using regex %s", regex_config["name"], regex_config["pattern"]) 
        if regex_config["pattern"] not in regex_cache:
            regex_cache[regex_config["pattern"]] = re.compile(regex_config["pattern"], re.IGNORECASE)
        result = regex_cache[regex_config["pattern"]].match(text)
        if result:
            metadata[regex_config["name"]] = result.group(0).strip()
        else:   
            logger.debug("No match found for regex %s in text", regex_config["pattern"])
    return metadata