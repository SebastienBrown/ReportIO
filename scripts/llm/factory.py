from .chat import chat_llm
from .embed import embed_llm

def get_llms():
    return {
        "chat": chat_llm,
        "embed": embed_llm
    }
