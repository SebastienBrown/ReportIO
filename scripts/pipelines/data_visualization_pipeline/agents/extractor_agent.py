from typing import List, TypedDict, Union
from flask import json
from langchain.output_parsers import StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from scripts.llm.chat import chat_llm


# -----------------------------
# Output schema from extractor
# -----------------------------
class ExtractedChartData(TypedDict):
    chart_type: str              # e.g., "bar", "line"
    title: str                   # chart title
    x_labels: List[str]          # categories, years, etc.
    y_values: List[Union[float, int]]
    units: List[str]             # e.g., "%", "jobs"
    notes: str                   # any assumptions or notes


# -----------------------------
# Output parser for the above schema
# -----------------------------
parser = StructuredOutputParser.from_response_schemas([
    {"name": "chart_type", "type": "string", "description": "Bar, line, pie, etc."},
    {"name": "title", "type": "string", "description": "Title of the chart"},
    {"name": "x_labels", "type": "array", "items": {"type": "string"}, "description": "Labels for the x-axis"},
    {"name": "y_values", "type": "array", "items": {"type": "number"}, "description": "Corresponding numerical values"},
    {"name": "units", "type": "array", "items": {"type": "string"}, "description": "Units for each y-value (e.g., %, $B, jobs)"},
    {"name": "notes", "type": "string", "description": "Any assumptions or context notes"}
])


# -----------------------------
# LLM Prompt for stat-to-chart extraction
# -----------------------------
EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a data extraction agent that converts numerical/statistical snippets into chart-ready structured data."),
    ("human", """
Extract structured data suitable for visualization from the following snippet:

"{snippet}"

Return chart type, title, x labels, y values, units, and any assumptions.

{format_instructions}
""")
]).partial(format_instructions=parser.get_format_instructions())


# -----------------------------
# Extractor Agent
# -----------------------------
def extractor_agent(state: dict) -> dict:
    extracted_docs = state.get("analysis_plan", {}).get("extracted_documents", [])
    chart_data_outputs = []

    print(f"[ExtractorAgent] Extracting structured data from {len(extracted_docs)} documents...")

    for doc in extracted_docs:
        if not doc.get("should_extract"):
            continue

        snippet = doc.get("snippet", "")
        try:
            result = (EXTRACTOR_PROMPT | chat_llm | parser).invoke({"snippet": snippet})
            chart_data_outputs.append({
                "document_index": doc["document_index"],
                "raw_snippet": snippet,
                "visualization_goal": doc["visualization_goal"],
                "reasoning": doc["reasoning"],
                "chart_data": result
            })
        except Exception as e:
            print(f"[ExtractorAgent] Failed to extract data from doc #{doc['document_index']}: {e}")

    state["extracted_chart_data"] = chart_data_outputs

    print(f"[ExtractorAgent] ✅ Extracted chart data from {len(chart_data_outputs)} / {len(extracted_docs)} documents")
    print("[ExtractorAgent] 📊 Sample Output:")
    if chart_data_outputs:
        print(json.dumps(chart_data_outputs[0], indent=2))

    return state
