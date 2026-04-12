from typing import Generator

from src.llm import ChatbotFactory


if __name__ == "__main__":
    chatbot = ChatbotFactory.create_chatbot(vendor="azure", stream_responses=False, batch_requests=False)
    response = chatbot.chat(
        [
            "What's the largest city by area in the world?",
            "What's the largest city by population in the world?",
        ]
    )
    for r in response:
        if isinstance(r, str):
            print(r)
        if isinstance(r, Generator):
            for chunk in r:
                print(chunk, end="", flush=True)
        print("\n=========================\n")
