from langchain_postgres.vectorstores import PGVector
from langchain_core.embeddings import Embeddings

from vector_store.vector_store import VectorStore


class PGVectorStore(VectorStore):
    def __init__(
        self,
        embeddings: Embeddings,
        connection_string: str,  # "postgresql+psycopg://user:pass@host:5432/db"
        collection_name: str = "documents",
    ):
        super().__init__(embeddings)
        self.connection_string = connection_string
        self.collection_name = collection_name
        self._store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True,  # store metadata as JSONB — faster filtering
        )

    def persist(self) -> None:
        pass  # pgvector commits on every add — nothing to do

    def clear(self) -> None:
        self._store.delete_collection()
