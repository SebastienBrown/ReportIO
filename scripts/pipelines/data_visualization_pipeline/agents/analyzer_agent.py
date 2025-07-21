from typing import List, TypedDict
from flask import json
from langchain.output_parsers import StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from scripts.llm.chat import chat_llm
from scripts.pipelines.data_visualization_pipeline import state


# -----------------------------
# Output schema per document
# -----------------------------
class AnalyzerDocOutput(TypedDict):
    document_index: int
    should_extract: bool
    should_project: bool
    visualization_goal: str
    reasoning: str
    snippet: str  # <- now LLM-extracted snippet only, not full doc


# -----------------------------
# Output parser
# -----------------------------
parser = StructuredOutputParser.from_response_schemas([
    {"name": "should_extract", "type": "boolean", "description": "True if document includes structured/numerical data worth extracting"},
    {"name": "should_project", "type": "boolean", "description": "True if projection (e.g. CAGR or filling years) is possible or needed"},
    {"name": "visualization_goal", "type": "string", "description": "What chart could be built from this document"},
    {"name": "reasoning", "type": "string", "description": "Explain what data was found and why it matters"},
    {"name": "snippet", "type": "string", "description": "Exact text snippet(s) from the document containing statistical or numerical information"},
    {"name": "document_index", "type": "integer", "description": "The index of the original document"}
])


# -----------------------------
# Prompt for LLM stat snippet extraction
# -----------------------------
ANALYZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a document analysis agent tasked with identifying relevant statistical or numerical data in web articles."),
    ("human", """
You are given a web document. Extract **only the most relevant text snippets** (one or more sentences, or short paragraph chunks) that contain **structured or numerical data** useful for visualizations or data analysis.

Return:
- The raw snippet(s) with numbers (copied exactly as-is)
- A short reasoning about what was found
- Whether the document is worth extracting (`should_extract`)
- Whether a projection is possible (`should_project`)
- What visualization could be built

Document #{document_index}:
-------------------
{document}
-------------------

{format_instructions}
""")
]).partial(format_instructions=parser.get_format_instructions())


# -----------------------------
# Analyzer agent
# -----------------------------
def analyzer_agent(state: dict) -> dict:
    all_docs = state.get("raw_documents", [])
    extracted_docs = []
    visualization_goals = []

    print(f"[AnalyzerAgent] Analyzing {len(all_docs)} documents individually...")

    for idx, doc in enumerate(all_docs):
        try:
            result = (ANALYZER_PROMPT | chat_llm | parser).invoke({
                "document": doc,
                "document_index": idx
            })
            extracted_docs.append(result)

            if result["should_extract"] and result["visualization_goal"]:
                visualization_goals.append(result["visualization_goal"])

        except Exception as e:
            print(f"[AnalyzerAgent] Failed on doc {idx}: {e}")

    # Filter by extractable
    filtered_docs = [d["snippet"] for d in extracted_docs if d["should_extract"]]

    state["analysis_plan"] = {
        "documents_analyzed": len(all_docs),
        "extracted_documents": extracted_docs,
        "visualization_goals": list(set(visualization_goals))
    }
    state["filtered_documents"] = filtered_docs

    print(f"[AnalyzerAgent] ✅ Extracted from {len(filtered_docs)} / {len(all_docs)} docs")
    print("[AnalyzerAgent] 🔎 Full Analysis Plan:")
    print(json.dumps(state["analysis_plan"], indent=2))
    return state
