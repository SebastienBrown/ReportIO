from scripts.llm.chat import chat_llm
from langchain.schema import SystemMessage, HumanMessage


def generate_answer_from_context(query: str, retrieved_chunks: list[str]) -> str:
    """Uses LLM to generate an answer based on the retrieved context"""
    
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the question clearly and concisely. "
        "If the answer is not found in the context, say so explicitly."
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
