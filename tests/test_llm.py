from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages.ai import AIMessage
from pydantic import BaseModel, Field

from src.llm import Chatbot, ChatbotFactory


# -- Test subclass that injects a mock LLM ----------------------------------


class FakeChatbot(Chatbot):
    def _build_llm(self):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="mock response")
        llm.stream.return_value = iter([AIMessage(content="chunk1"), AIMessage(content="chunk2")])
        llm.batch.return_value = [AIMessage(content="resp1"), AIMessage(content="resp2")]
        return llm


@pytest.fixture
def bot():
    return FakeChatbot(stream_responses=False, batch_requests=False)


@pytest.fixture
def streaming_bot():
    return FakeChatbot(stream_responses=True, batch_requests=False)


@pytest.fixture
def batch_bot():
    return FakeChatbot(stream_responses=False, batch_requests=True)


# -- Subclass contract -------------------------------------------------------


class TestSubclassContract:
    def test_base_chatbot_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Chatbot()

    def test_factory_raises_on_unknown_vendor(self):
        with pytest.raises(ValueError, match="Unsupported vendor"):
            ChatbotFactory.create_chatbot(vendor="unknown")

    def test_subclass_sets_base_llm_and_chain(self, bot):
        assert bot.base_llm is not None
        assert bot.llm is not None


# -- _run dispatch -----------------------------------------------------------


class TestRunDispatch:
    def test_invoke_when_not_streaming(self, bot):
        chain = MagicMock()
        chain.invoke.return_value = "invoked"
        result = bot._run(chain, "hello")
        chain.invoke.assert_called_once_with("hello")
        assert result == "invoked"

    def test_stream_when_streaming(self, streaming_bot):
        chain = MagicMock()
        chain.stream.return_value = iter(["a", "b"])
        result = streaming_bot._run(chain, "hello")
        chain.stream.assert_called_once_with("hello")
        assert list(result) == ["a", "b"]

    def test_set_streaming_switches_mode(self, bot):
        chain = MagicMock()
        chain.invoke.return_value = "invoked"
        chain.stream.return_value = iter(["streamed"])

        bot._run(chain, "x")
        chain.invoke.assert_called_once()

        bot.set_streaming(True)
        result = bot._run(chain, "x")
        chain.stream.assert_called_once()
        assert list(result) == ["streamed"]


# -- chat --------------------------------------------------------------------


class TestChat:
    def test_chat_batches_when_enabled(self, batch_bot):
        batch_bot.llm = MagicMock()
        batch_bot.llm.batch.return_value = ["r1", "r2"]
        result = batch_bot.chat(["q1", "q2"])
        batch_bot.llm.batch.assert_called_once_with(["q1", "q2"])
        assert result == ["r1", "r2"]

    def test_chat_iterates_when_batch_disabled(self, bot):
        bot.llm = MagicMock()
        bot.llm.invoke.side_effect = ["r1", "r2"]
        result = bot.chat(["q1", "q2"])
        assert bot.llm.invoke.call_count == 2
        assert result == ["r1", "r2"]


# -- explain_topic / retrieve_topic_info chain wiring ------------------------


class TestChainMethods:
    def test_explain_topic_invokes_chain(self, bot):
        with patch.object(bot, "_run", return_value="explanation") as mock_run:
            result = bot.explain_topic(topic="gravity", role="teacher", level="simple")
            assert result == "explanation"
            input_dict = mock_run.call_args[0][1]
            assert input_dict == {"role": "teacher", "topic": "gravity", "level": "simple"}

    def test_retrieve_topic_info_invokes_chain(self, bot):
        class Planet(BaseModel):
            name: str = Field(description="Planet name")
            moons: int = Field(description="Number of moons")

        with patch.object(bot, "_run", return_value=Planet(name="Mars", moons=2)) as mock_run:
            result = bot.retrieve_topic_info(model=Planet, topic="Mars", text="")
            assert result.name == "Mars"
            input_dict = mock_run.call_args[0][1]
            assert input_dict["topic"] == "Mars"
            assert input_dict["known"] == "none"
