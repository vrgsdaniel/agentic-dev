from src.llm import ChatbotFactory

from utils import TextProcessor, print_stream
from vector_store.embeddings import EmbeddingsFactory
from vector_store.vector_store import VectorStoreFactory

FULLY_LOAD = False

if __name__ == "__main__":
    if FULLY_LOAD:
        text_processor = TextProcessor(chunk_size=500, chunk_overlap=100)
        embeddings = EmbeddingsFactory.create_embeddings(vendor="azure")
        vector_store = VectorStoreFactory.create_(vendor="chroma", embeddings=embeddings)
        chatbot = ChatbotFactory.create_chatbot(vendor="azure", stream_responses=True, batch_requests=False)

        documents = text_processor.load("docs/hnsw.pdf")
        vector_store.add(text_processor.split(documents=documents))

        chatbot.set_rag_chain(vector_store.as_retriever(search_kwargs={"k": 6}))
    else:
        embeddings = EmbeddingsFactory.create_embeddings(vendor="azure")
        vector_store = VectorStoreFactory.create_(vendor="chroma", embeddings=embeddings)
        chatbot = ChatbotFactory.create_chatbot(vendor="azure", stream_responses=True, batch_requests=False)
        chatbot.set_rag_chain(vector_store.as_retriever(search_kwargs={"k": 6}))

        response = chatbot.ask_with_context("What is a barycentre?")
        print_stream(response)
