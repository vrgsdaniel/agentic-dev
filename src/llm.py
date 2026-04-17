from typing import List

from langsmith import traceable
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser,
    JsonOutputParser,
)
from langchain_classic.output_parsers import OutputFixingParser
from pydantic import BaseModel

from src.prompts import EXPLAIN_TOPIC_PROMPT, FETCH_TOPIC_PROMPT, RETRIEVE_FROM_CONTEXT_PROMPT
from src.settings import AzureLLMSettings
from langchain_core.retrievers import BaseRetriever


class Chatbot:
    def __init__(
        self,
        stream_responses: bool = False,
        batch_requests: bool = True,
    ):
        self.stream_responses = stream_responses
        self.batch_requests = batch_requests
        self.base_llm = self._build_llm()
        self.parser = self.get_parser("str")
        self.llm = self.base_llm | self.parser
        self.rag_chain: Runnable | None = None

    def get_parser(self, parser_type: str, pydantic_model: type[BaseModel] | None = None):
        if parser_type == "str":
            return StrOutputParser()
        if parser_type == "json":
            return JsonOutputParser()
        if parser_type == "pydantic":
            if pydantic_model is None:
                raise ValueError("pydantic_model is required for parser_type='pydantic'")
            return PydanticOutputParser(pydantic_object=pydantic_model)
        raise ValueError(f"Unsupported parser type: {parser_type}")

    def _build_llm(self) -> BaseChatModel:
        """
        Intended to be overridden by subclasses to construct the appropriate LLM instance based on settings.
        """
        raise NotImplementedError("Subclasses must implement _build_llm() to return an LLM instance.")

    @traceable(name="chain-execution")
    def _run(self, chain, input):
        if self.stream_responses:
            return chain.stream(input)
        return chain.invoke(input)

    # interface methods for chat

    def set_streaming(self, stream: bool):
        self.stream_responses = stream

    def chat(self, prompts: List[str]) -> List[str]:
        if self.batch_requests:
            return self.llm.batch(prompts)
        return [self._run(self.llm, prompt) for prompt in prompts]

    def explain_topic(self, topic: str, role: str = "helpful assistant", level: str = "simple") -> str:
        chain = EXPLAIN_TOPIC_PROMPT | self.llm
        return self._run(chain, {"role": role, "topic": topic, "level": level})

    def fetch_topic_details(self, model: type[BaseModel], topic: str, text: str) -> dict:
        parser = self.get_parser("pydantic", pydantic_model=model)
        fields_text = "\n".join(f"- {name}: {info.description}" for name, info in model.model_fields.items())
        prompt = FETCH_TOPIC_PROMPT.partial(fields=fields_text, format_instructions=parser.get_format_instructions())
        fixing_parser = OutputFixingParser.from_llm(
            parser=parser,
            llm=self.base_llm,
        )

        chain = prompt | self.base_llm | fixing_parser
        return self._run(chain, {"topic": topic, "known": text or "none"})

    # Interface methods for retrieval-augmented generation (RAG)
    def set_rag_chain(self, retriever: BaseRetriever):
        self.rag_chain = (
            RunnableParallel(
                {
                    "context": retriever | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)),
                    "question": RunnablePassthrough(),
                }
            )
            | RETRIEVE_FROM_CONTEXT_PROMPT
            | self.llm
        )

    def ask_with_context(self, question: str) -> str:
        if self.rag_chain is None:
            raise ValueError("RAG chain not set. Call set_rag_chain() after setting a retriever.")
        return self._run(self.rag_chain, question)


class AzureChatbot(Chatbot):
    def __init__(self, stream_responses: bool = False, batch_requests: bool = True):
        self._settings = AzureLLMSettings()
        super().__init__(stream_responses=stream_responses, batch_requests=batch_requests)

    def _build_llm(self) -> BaseChatModel:
        """
        Construct an ``AzureChatOpenAI`` instance from the current settings.
        """
        return AzureChatOpenAI(
            azure_endpoint=self._settings.azure_openai_endpoint,
            azure_deployment=self._settings.azure_deployment,
            openai_api_version=self._settings.open_ai_version,
            api_key=self._settings.azure_openai_api_key.get_secret_value(),
            temperature=self._settings.temperature,
            max_retries=self._settings.max_retries,
        )


class ChatbotFactory:
    @staticmethod
    def create_chatbot(vendor: str, stream_responses: bool = False, batch_requests: bool = True) -> Chatbot:
        if vendor == "azure":
            return AzureChatbot(stream_responses=stream_responses, batch_requests=batch_requests)
        # Add more vendors here as needed
        raise ValueError(f"Unsupported vendor: {vendor}")
