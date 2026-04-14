from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStore:
    def __init__(
        self,
        embeddings: Embeddings,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwargs,
    ):
        self.embeddings = embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._store = None

    def as_retriever(self, **kwargs) -> BaseRetriever:
        return self._store.as_retriever(**kwargs)

    def add(self, documents: List[Document]) -> None:
        self._store.add_documents(documents)

    def load(self, source: str) -> List[Document]:
        loader = PyPDFLoader(source) if source.endswith(".pdf") else TextLoader(source)
        return loader.load()

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)

    def persist(self) -> None:
        """Persist the store to disk if supported."""
        raise NotImplementedError

    def clear(self) -> None:
        """Delete all documents from the store."""
        raise NotImplementedError

    def ingest(self, source: str) -> None:
        """Full pipeline: load → split → add."""
        docs = self.load(source)
        chunks = self.split(docs)
        self.add(chunks)


class VectorStoreFactory:
    @staticmethod
    def create_(vendor: str, embeddings: Embeddings, **kwargs) -> VectorStore:
        if vendor == "chroma":
            from vector_store.chroma_vector_store import ChromaVectorStore

            return ChromaVectorStore(embeddings, **kwargs)
        if vendor == "pgvector":
            from vector_store.pg_vector_store import PGVectorStore

            return PGVectorStore(embeddings, **kwargs)
        # Add more vendors here as needed
        raise ValueError(f"Unsupported vendor: {vendor}")
