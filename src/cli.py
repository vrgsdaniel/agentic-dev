from typing import List

from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.prompt import Prompt

from src.llm import ChatbotFactory
from src.utils import TextProcessor, print_stream
from src.vector_store.embeddings import EmbeddingsFactory
from src.vector_store.vector_store import VectorStoreFactory

console = Console()


class Planet(BaseModel):
    name: str = Field(description="The name of the planet")
    type: str = Field(description="The type of the planet (e.g., terrestrial, gas giant)")
    distance_from_sun: float = Field(description="Distance from the sun in million kilometers")
    number_of_moons: int = Field(description="Number of moons orbiting the planet")
    elements: List[str] = Field(description="List of key elements found on the planet")
    name_of_largest_moon: str | None = Field(None, description="The name of the largest moon orbiting the planet")
    cool_facts: str = Field(description="One sentence with any cool facts about the planet")

    @field_validator("cool_facts")
    def one_sentence(cls, v):
        if v.count(".") > 1:
            raise ValueError("summary must be a single sentence")
        return v.strip()


MODES = {
    "1": "Chat",
    "2": "Explain",
    "3": "Extract",
    "4": "RAG",
}


def show_menu():
    console.print("\n[bold cyan]Modes[/bold cyan]")
    for key, name in MODES.items():
        console.print(f"  [bold]{key}[/bold] - {name}")
    console.print("  [bold]q[/bold] - Quit")


def chat_mode(chatbot):
    console.print("\n[bold green]Chat mode[/bold green] \u2014 type your message, /back to return\n")
    while True:
        user_input = Prompt.ask("[bold]You[/bold]")
        if user_input.strip().lower() == "/back":
            break
        console.print("[bold magenta]Bot[/bold magenta]", end=" ")
        response = chatbot._run(chatbot.llm, user_input)
        if chatbot.stream_responses:
            print_stream(response)
        else:
            console.print(response)
        console.print()


def explain_mode(chatbot):
    console.print("\n[bold green]Explain mode[/bold green] \u2014 explain any topic with a role and level\n")
    while True:
        topic = Prompt.ask("[bold]Topic[/bold] (/back to return)")
        if topic.strip().lower() == "/back":
            break
        role = Prompt.ask("[bold]Role[/bold]", default="helpful assistant")
        level = Prompt.ask("[bold]Level[/bold]", default="simple")

        console.print("[bold magenta]Bot[/bold magenta]", end=" ")
        response = chatbot.explain_topic(topic=topic, role=role, level=level)
        if chatbot.stream_responses:
            print_stream(response)
        else:
            console.print(response)
        console.print()


def extract_mode(chatbot):
    console.print("\n[bold green]Extract mode[/bold green]" " \u2014 extract structured Planet data from the LLM\n")
    was_streaming = chatbot.stream_responses
    chatbot.set_streaming(False)

    while True:
        topic = Prompt.ask("[bold]Planet name[/bold] (/back to return)")
        if topic.strip().lower() == "/back":
            break
        known = Prompt.ask("[bold]Known facts[/bold] (optional, press Enter to skip)", default="")

        console.print("[dim]Extracting\u2026[/dim]")
        result = chatbot.fetch_topic_details(model=Planet, topic=topic, text=known)
        console.print(result)
        console.print()

    chatbot.set_streaming(was_streaming)


def rag_mode(chatbot):
    console.print(
        "\n[bold green]RAG mode[/bold green]"
        " \u2014 ask questions against loaded documents\n"
        "  [bold]/load[/bold] <path>  \u2014 load a PDF or text file\n"
        "  [bold]/back[/bold]         \u2014 return to menu\n"
    )

    text_processor = TextProcessor(chunk_size=500, chunk_overlap=100)
    embeddings = EmbeddingsFactory.create_embeddings(vendor="azure")
    vector_store = VectorStoreFactory.create_vector_store(vendor="chroma", embeddings=embeddings)
    total_chunks = 0

    # Connect to existing persisted store if available
    chatbot.set_rag_chain(vector_store.as_retriever(search_kwargs={"k": 6}))
    console.print("[dim]Connected to existing vector store.[/dim]")

    while True:
        user_input = Prompt.ask("[bold]RAG[/bold]")
        stripped = user_input.strip()

        if stripped.lower() == "/back":
            break

        if stripped.lower().startswith("/load"):
            file_path = stripped[5:].strip()
            if not file_path:
                console.print("[red]Usage: /load <path>[/red]")
                continue

            console.print("[dim]Loading and indexing document\u2026[/dim]")
            documents = text_processor.load(file_path)
            chunks = text_processor.split(documents)
            vector_store.add(chunks)
            total_chunks += len(chunks)

            chatbot.set_rag_chain(vector_store.as_retriever(search_kwargs={"k": 6}))
            console.print(f"[dim]Indexed {len(chunks)} chunks ({total_chunks} new). Ask away![/dim]\n")
            continue

        console.print("[bold magenta]Bot[/bold magenta]", end=" ")
        response = chatbot.ask_with_context(stripped)
        if chatbot.stream_responses:
            print_stream(response)
        else:
            console.print(response)
        console.print()


def main():
    console.print("[bold cyan]agentic-dev CLI[/bold cyan]")
    chatbot = ChatbotFactory.create_chatbot(vendor="azure", stream_responses=True, batch_requests=False)
    console.print("[dim]Chatbot ready.[/dim]")

    mode_handlers = {
        "1": chat_mode,
        "2": explain_mode,
        "3": extract_mode,
        "4": rag_mode,
    }

    while True:
        show_menu()
        choice = Prompt.ask("\n[bold]Select mode[/bold]")
        if choice.strip().lower() == "q":
            console.print("[bold]Bye![/bold]")
            break
        handler = mode_handlers.get(choice)
        if handler:
            handler(chatbot)
        else:
            console.print("[red]Invalid choice.[/red]")


if __name__ == "__main__":
    main()
