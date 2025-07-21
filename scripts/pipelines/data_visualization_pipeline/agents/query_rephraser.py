from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser
from scripts.llm.chat import chat_llm


# 1. Define output structure for clarity
class RephraserOutput(TypedDict):
    subqueries: List[str]


# 2. Structured parser
parser = StructuredOutputParser.from_response_schemas([
    {
        "name": "subqueries",
        "description": "List of 1–3 short subqueries that can help retrieve structured data from the web",
        "type": "list[str]"
    }
])


# 3. Prompt with auto format instructions
REPHRASER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that rewrites user questions to retrieve data from the web."),
    ("human", """Break down the user's question into 1–3 focused subqueries that are likely to return factual or quantitative data — such as statistics, trends, forecasts, numerical comparisons, pricing, market data, weather, rankings, performance metrics, or scientific measurements.

Avoid general explanations, definitions, or how-to phrasing. Focus on getting queries that are **data-rich** and **fact-oriented**.

Query: {user_query}

{format_instructions}
""")
]).partial(format_instructions=parser.get_format_instructions())


# 4. Chain: prompt → chat model → parser LCEL
rephraser_chain = REPHRASER_PROMPT | chat_llm | parser


# 5. LangGraph-compatible node
def query_rephraser(state: dict) -> dict:
    result = rephraser_chain.invoke({"user_query": state["user_query"]})
    state["rephrased_queries"] = result["subqueries"]
    return state
