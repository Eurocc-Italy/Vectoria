#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from vectoria_lib.common.paths import ETC_DIR

def format_docs(docs):
    """
    Helper function to format the retrieved documents for use in the context.

    Parameters:
    - docs (list): A list of document objects containing content.

    Returns:
    - str: Formatted string with concatenated page content from the documents.
    """
    
    return "\n\n".join(doc.page_content for doc in docs)

def get_prompt(file_name: str, lang):
    if lang not in ["it", "eng"]:
        raise ValueError(f"lang '{lang}' is not valid. Supported: 'it', 'eng'.")
    custom_prompt = ETC_DIR / "custom" / "prompts" / lang / file_name
    if custom_prompt.exists():
        return custom_prompt.read_text()
    else:
        return (ETC_DIR / "default" / "prompts" / lang / file_name).read_text()