from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from vector_store.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str = "./chroma_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        super().__init__(embeddings, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.persist_directory = persist_directory
        self._store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )

    def persist(self) -> None:
        self._store.persist()

    def clear(self) -> None:
        self._store.delete_collection()
