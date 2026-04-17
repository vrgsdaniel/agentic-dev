from src.llm import ChatbotFactory
from src.utils import print_stream


if __name__ == "__main__":
    chatbot = ChatbotFactory.create_chatbot(vendor="azure", stream_responses=False, batch_requests=False)
    response = chatbot.explain_topic(
        topic="heavy water reactors",
        role="physics professor",
        level="elementary school",
    )
    print_stream(response)

    print("\n=========================\n")

    chatbot.set_streaming(True)
    response = chatbot.explain_topic(
        topic="partial objects in Python",
        role="Python expert",
        level="simple, concise, examples-based",
    )
    print_stream(response)
