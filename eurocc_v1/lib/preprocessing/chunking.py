from langchain.text_splitter import RecursiveCharacterTextSplitter

def make_chunks(text: str, chunk_size: int = 1024, chunk_overlap: int = 256) -> list[str]:
    # This text splitter is the recommended one for generic text. It is parameterized by a list of characters. 
    # It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. 
    # This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, 
    # as those would generically seem to be the strongest semantically related pieces of text.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.create_documents([text])
    