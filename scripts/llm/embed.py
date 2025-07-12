import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

embed_llm = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    openai_api_version=os.environ.get("CHATOPENAI_API_VERSION", "2023-05-15"),  # adjust if needed
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

__all__ = ["embed_llm"]
