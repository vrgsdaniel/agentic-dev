from langchain_core.prompts import ChatPromptTemplate

EXPLAIN_TOPIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {role}. Be concise."),
        ("human", "Explain {topic} in {level} terms."),
    ]
)

FETCH_TOPIC_PROMPT = ChatPromptTemplate.from_messages(
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


RETRIEVE_FROM_CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only the context below. "
            "If the answer isn't in the context, say so.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)
