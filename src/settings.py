from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureLLMSettings(BaseSettings):
    """Azure OpenAI LLM configuration loaded from environment / dotenv file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    env: str = "dev"
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    temperature: float = 0.7
    max_retries: int = 3

    @computed_field
    @property
    def azure_deployment(self) -> str:
        return (
            "Entity_Extractor_GPT4.1_TEST"
            if self.env == "prd"
            else "Document_Extractor_GPT4.1_TEST"
        )

    @computed_field
    @property
    def open_ai_version(self) -> str:
        return "2024-12-01-preview" if self.env == "prd" else "2025-01-01-preview"
