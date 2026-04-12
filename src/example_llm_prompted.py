from src.llm import ChatbotFactory


if __name__ == "__main__":
    chatbot = ChatbotFactory.create_chatbot(
        vendor="azure", stream_responses=False, batch_requests=False
    )
    response = chatbot.explain_topic(
        topic="heavy water reactors",
        role="physics professor",
        level="elementary school",
    )
    for chunk in response:
        print(chunk, end="", flush=True)

    print("\n=========================\n")

    chatbot.set_streaming(True)
    response = chatbot.explain_topic(
        topic="partial objects in Python",
        role="Python expert",
        level="simple, concise, examples-based",
    )
    for chunk in response:
        print(chunk, end="", flush=True)
