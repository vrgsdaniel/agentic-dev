from src.llm import ChatbotFactory


if __name__ == "__main__":
    chatbot = ChatbotFactory.create_chatbot(vendor="azure", stream_responses=True, batch_requests=False)
    response = chatbot.explain_topic(
        topic="heavy water reactors",
        role="physics professor",
        level="elementary school",
    )
    for chunk in response:
        print(chunk, end="", flush=True)
