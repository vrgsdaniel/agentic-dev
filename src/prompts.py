from langchain_core.prompts import ChatPromptTemplate

EXPLAIN_TOPIC = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {role}. Be concise."),
        ("human", "Explain {topic} in {level} terms."),
    ]
)

RETRIEVE_INFO = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable assistant.\n"
            "Fill in the following fields for the given topic:\n{fields}\n\n"
            "The user may provide some known values — treat them as ground truth "
            "and only infer the remaining fields.\n\n"
            "{format_instructions}",
        ),
        ("human", "Topic: {topic}\nKnown values: {known}"),
    ]
)
