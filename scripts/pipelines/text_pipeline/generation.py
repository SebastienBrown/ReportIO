from scripts.llm.chat import chat_llm
from langchain.schema import SystemMessage, HumanMessage


def generate_answer_from_context(query: str, retrieved_chunks: list[str]) -> str:
    """Uses LLM to generate an answer based on the retrieved context"""
    
    system_prompt = (
    "You are an expert research assistant and technical report writer. Your task is to write a long-form, structured chapter "
    "on a specific subtopic of a larger research question. You will be given a chapter title and a collection of context documents, "
    "each with source identifiers (e.g., [SOURCE 1], [SOURCE 2]).\n\n"

    "Your chapter must:\n"
    "- Begin with a concise introduction (2–3 sentences) that previews what the section will cover.\n"
    "- Contain 2 to 3 subsections (with subheadings) that break down the topic logically and in depth.\n"
    "- In each subsection:\n"
    "  • Use evidence, examples, or data from the provided sources.\n"
    "  • Clearly cite the source(s) in brackets like [SOURCE 1] next to the claims.\n"
    "  • If multiple sources support the same point, cite them together, e.g., [SOURCE 2][SOURCE 4].\n"
    "  • Compare differing viewpoints if available, and highlight gaps where applicable.\n"
    "- Avoid adding information that isn’t grounded in the provided sources.\n"
    "- End the chapter with a short summary or takeaway paragraph (2–4 sentences).\n\n"

    "Formatting:\n"
    "- Use markdown-style formatting:\n"
    "  • Chapter title as H2 (##)\n"
    "  • Subsection titles as H3 (###)\n"
    "  • Paragraphs separated by double line breaks\n"
    "- Keep tone factual, analytical, and accessible to a general expert audience.\n\n"

    "If the provided context lacks sufficient information to cover the chapter title, say:\n"
    "'The provided context does not contain enough relevant information to write this section.'\n"
    )

    
    context = "\n\n".join(retrieved_chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    response = chat_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    return response.content

def generate_gpt_answer(query: str) -> str:
    """Generates a direct GPT response without RAG context."""
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer clearly and concisely."),
        HumanMessage(content=query)
    ]
    response = chat_llm.invoke(messages)
    return response.content


__all__ = ["chat_llm", "generate_gpt_answer"]
