from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from src.vector_store.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str = "./chroma_db",
    ):
        super().__init__(embeddings)
        self.persist_directory = persist_directory
        self._store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )

    def persist(self) -> None:
        self._store.persist()

    def clear(self) -> None:
        self._store.delete_collection()
