from langchain.embeddings import Embeddings

from src.settings import AzureLLMSettings


class EmbeddingsFactory:
    @staticmethod
    def create_embeddings(vendor: str, model: str = "text-embedding-3-small") -> Embeddings:
        if vendor == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        if vendor == "azure":
            from langchain_openai import AzureOpenAIEmbeddings

            settings = AzureLLMSettings()

            return AzureOpenAIEmbeddings(
                model=settings.azure_embeddings_deployment,
                deployment=settings.azure_embeddings_deployment,
                api_version=settings.azure_embeddings_version,
            )
        raise ValueError(f"Unsupported embeddings vendor: {vendor}")
