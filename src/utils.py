from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rich import print as rprint


def print_stream(stream):
    for chunk in stream:
        rprint(chunk, end="", flush=True)


class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load(self, source: str) -> List[Document]:
        loader = PyPDFLoader(source) if source.endswith(".pdf") else TextLoader(source)
        return loader.load()

    def split(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)
