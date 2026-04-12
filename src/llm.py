from typing import List

from langchain_openai import AzureChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser, JsonOutputParser

from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.settings import AzureLLMSettings
from typing import Generator


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
        self.llm = self.build_chain(self.base_llm, self.parser)
        self.explain_topic_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a {role}. Be concise."),
                ("human", "Explain {topic} in {level} terms."),
            ]
        )
        self.retrieve_info_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a knowledgeable assistant.\n"
                    "Fill in the following fields for the given topic:\n{fields}\n\n"
                    "The user may provide some known values — treat them as ground truth "
                    "and only infer the remaining fields.\n\n"
                    "{format_instructions}",
                ),
                ("human", "Topic: {topic}\n" "Known values: {known}"),
            ]
        )

    def build_chain(self, model, parser):
        return model | parser

    def get_parser(self, parser_type: str) -> StrOutputParser:
        if parser_type == "str":
            return StrOutputParser()
        if parser_type == "json":
            return JsonOutputParser()
        # Add more parser types here as needed
        raise ValueError(f"Unsupported parser type: {parser_type}")

    def _build_llm(self) -> AzureChatOpenAI:
        """
        Intended to be overridden by subclasses to construct the appropriate LLM instance based on settings.
        """
        ...

    @staticmethod
    def _invoke(llm: BaseChatModel, prompt: str) -> AIMessage:
        response = llm.invoke(prompt)
        return response

    @staticmethod
    def _stream(llm: BaseChatModel, prompt: str) -> Generator[AIMessage, None, None]:
        for chunk in llm.stream(prompt):
            response = chunk
            yield response

    @staticmethod
    def _batch(llm: BaseChatModel, prompts: List[str]) -> List[AIMessage]:
        response = llm.batch(prompts)
        return [r for r in response]

    def answer(self, llm: BaseChatModel, question: str) -> str:
        if llm is None:
            llm = self.llm
        response = self._invoke(llm, question) if not self.stream_responses else self._stream(llm, question)
        return response

    def chat(self, prompts: List[str]) -> List[str]:
        if self.batch_requests:
            responses = self._batch(self.llm, prompts)
            return responses

        responses = []
        for prompt in prompts:
            response = self.answer(self.llm, prompt)
            responses.append(response)
        return responses

    def explain_topic(self, topic: str, role: str = "helpful assistant", level: str = "simple") -> str:
        chain = self.explain_topic_prompt | self.llm
        response = chain.invoke({"role": role, "topic": topic, "level": level})
        return response

    def retrieve_topic_info(self, model: type[BaseModel], topic: str, text: str) -> dict:
        parser = PydanticOutputParser(pydantic_object=model)
        fields_text = "\n".join(f"- {name}: {info.description}" for name, info in model.model_fields.items())
        prompt = self.retrieve_info_prompt.partial(
            fields=fields_text, format_instructions=parser.get_format_instructions()
        )
        llm = prompt | self.base_llm | parser
        response = self.answer(
            llm,
            {
                "topic": topic,
                "known": text or "none",
            },
        )

        return response


class AzureChatbot(Chatbot):
    def __init__(self, stream_responses: bool = False, batch_requests: bool = True):
        self._settings = AzureLLMSettings()
        super().__init__(stream_responses=stream_responses, batch_requests=batch_requests)

    def _build_llm(self) -> AzureChatOpenAI:
        """
        Construct an ``AzureChatOpenAI`` instance from the current settings.
        """
        return AzureChatOpenAI(
            azure_endpoint=self._settings.azure_openai_endpoint,
            azure_deployment=self._settings.azure_deployment,
            openai_api_version=self._settings.open_ai_version,
            api_key=self._settings.azure_openai_api_key,
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
