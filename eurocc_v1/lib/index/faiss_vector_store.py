from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class FaissVectorStore:
    # https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/


    @staticmethod
    def load_from_pickle(model_name: str, pkl: bytes):
        fvs = FaissVectorStore(model_name)
        index = FAISS.deserialize_from_bytes(
            embeddings=fvs.hf_embedder, serialized=pkl, allow_dangerous_deserialization=True
        )
        fvs.index = index
        return fvs

    def __init__(self, model_name, device="cuda"):
        self.model_name = model_name
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": False}
        self.hf_embedder = HuggingFaceBgeEmbeddings(
            model_name=self.model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        self.index = None

    def make_index(self, docs: list[str]):
        self.index = FAISS.from_documents(docs, self.hf_embedder)
        return self

    def as_retriever(self, **kwargs):
        if self.index is None:
            raise ValueError("Index is not created. Call make_index() first.")
            
        return self.index.as_retriever(**kwargs)

    def dump_to_pickle(self, output_path: str | Path = ".") -> Path:
        pkl = self.index.serialize_to_bytes()
        model_name = self.model_name.replace('/','__') # repo/name => repo__name
        pkl_path = Path(output_path) / f"{model_name}_faiss_index.pkl"
        with open(pkl_path, "wb") as f:
            f.write(pkl)
        return pkl_path

        
