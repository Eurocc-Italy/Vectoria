import os
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# extract_text_from_pdf(DATA_DIR / "raw")

def extract_text_from_pdf(pdf_folder_path: Path) -> str:

    logger = logging.getLogger('ecclogger')

    loaders = [
        PyPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)
    ]
    pages = loaders[0].load() # FIXME: using just one single doc for testing

    pages_str = ""
    pages_list = []
    for page in pages:
        pages_list.append(page.page_content)

    pages_str = "".join(pages_list)
    logger.debug(f"Loaded {len(pages_str)} characters")
    
    return pages_str