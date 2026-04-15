from langchain.embeddings import Embeddings

from src.settings import AzureLLMSettings


class EmbeddingsFactory:
    @staticmethod
    def create_embeddings(vendor: str) -> Embeddings:
        if vendor == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model="text-embedding-3-small")
        if vendor == "azure":
            from langchain_openai import AzureOpenAIEmbeddings

            settings = AzureLLMSettings()

            return AzureOpenAIEmbeddings(
                model="text-embedding-3-small",
                deployment=settings.azure_embeddings_deployment,
                api_version=settings.azure_embeddings_version,
            )
        else:
            raise ValueError(f"Unsupported embeddings vendor: {vendor}")
