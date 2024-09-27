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

def get_prompt():
    custom_prompt = ETC_DIR / "custom" / "qa_prompt.txt"
    if custom_prompt.exists():
        return custom_prompt.read_text()
    else:
        return (ETC_DIR / "default" / "qa_prompt.txt").read_text()