from typing import List

from pydantic import BaseModel, Field
from src.llm import ChatbotFactory
from rich import print as rprint


class Planet(BaseModel):
    name: str = Field(description="The name of the planet")
    type: str = Field(description="The type of the planet (e.g., terrestrial, gas giant)")
    distance_from_sun: float = Field(description="Distance from the sun in million kilometers")
    number_of_moons: int = Field(description="Number of moons orbiting the planet")
    elements: List[str] = Field(description="List of key elements found on the planet")
    cool_facts: str = Field(description="Any cool facts about the planet")


if __name__ == "__main__":
    chatbot = ChatbotFactory.create_chatbot(vendor="azure", stream_responses=False, batch_requests=False)
    response = chatbot.retrieve_topic_info(model=Planet, topic="Jupiter", text="")
    for chunk in response:
        rprint(chunk, end="", flush=True)

    response = chatbot.retrieve_topic_info(
        model=Planet,
        topic="Neptune",
        text="it's a cold ice giant, with 2 moons, and a cool fact is that is has supersonic winds",  # 2 moons is incorrect, but let's see how the model handles it
    )
    for chunk in response:
        rprint(chunk, end="", flush=True)
