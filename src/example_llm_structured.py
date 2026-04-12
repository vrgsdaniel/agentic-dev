from typing import List

from pydantic import BaseModel, Field, field_validator
from src.llm import ChatbotFactory
from rich import print as rprint


class Planet(BaseModel):
    name: str = Field(description="The name of the planet")
    type: str = Field(
        description="The type of the planet (e.g., terrestrial, gas giant)"
    )
    distance_from_sun: float = Field(
        description="Distance from the sun in million kilometers"
    )
    number_of_moons: int = Field(description="Number of moons orbiting the planet")
    elements: List[str] = Field(description="List of key elements found on the planet")
    # Optional field to demonstrate handling of missing information
    name_of_largest_moon: str | None = Field(
        None, description="The name of the largest moon orbiting the planet"
    )
    cool_facts: str = Field(
        description="One sentence with any cool facts about the planet"
    )

    @field_validator("cool_facts")
    def one_sentence(cls, v):
        if v.count(".") > 1:
            raise ValueError("summary must be a single sentence")
        return v.strip()


if __name__ == "__main__":
    chatbot = ChatbotFactory.create_chatbot(
        vendor="azure", stream_responses=False, batch_requests=False
    )
    response = chatbot.retrieve_topic_info(
        model=Planet,
        topic="Venus",
        text="",
    )
    for chunk in response:
        rprint(chunk, end="", flush=True)

    response = chatbot.retrieve_topic_info(
        model=Planet,
        topic="Neptune",
        # 2 moons is incorrect, but let's see how the model handles it
        text="it's a cold ice giant, with 2 moons, and a cool fact is that is has supersonic winds",
    )
    for chunk in response:
        rprint(chunk, end="", flush=True)
