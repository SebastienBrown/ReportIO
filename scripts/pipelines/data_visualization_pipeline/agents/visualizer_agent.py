from typing import List, TypedDict, Union
import json
import os
import uuid
import traceback

from langchain.output_parsers import StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from scripts.llm.chat import chat_llm


# -----------------------------
# Input structure (from extractor)
# -----------------------------
class ChartData(TypedDict):
    chart_type: str
    title: str
    x_labels: List[str]
    y_values: List[Union[float, int]]
    units: List[str]
    notes: str


# -----------------------------
# Prompt for LLM to generate matplotlib code
# -----------------------------
PYTHON_CODE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a Python charting assistant. You write clean matplotlib code to generate charts as PNG files."),
    ("human", """
Write a Python script using matplotlib that generates a {chart_type} chart.

Use the following data:
- Title: {title}
- X labels: {x_labels}
- Y values: {y_values}
- Units: {units}

Instructions:
- Save the chart as 'chart.png'
- Label axes and use readable ticks
- Don't call plt.show(); only save the file

Return only the Python code in a markdown code block.
""")
])


# -----------------------------
# Render code to PNG (sandboxed exec)
# -----------------------------
def run_python_code_and_save_image(code_str: str, output_dir: str = "charts") -> str:
    os.makedirs(output_dir, exist_ok=True)
    unique_filename = f"{uuid.uuid4().hex}.png"
    file_path = os.path.join(output_dir, unique_filename)

    # Replace any hardcoded filename with our actual path
    code_str = code_str.replace("chart.png", file_path)

    try:
        exec_globals = {}
        exec(code_str, exec_globals)
        return file_path
    except Exception as e:
        print("[VisualizerAgent] ❌ Chart rendering failed:")
        traceback.print_exc()
        return None


# -----------------------------
# Visualizer Agent
# -----------------------------
def visualizer_agent(state: dict) -> dict:
    chart_data_list = state.get("extracted_chart_data", [])
    visualizations = []

    print(f"[VisualizerAgent] 🧠 Generating code & rendering {len(chart_data_list)} charts...")

    for item in chart_data_list:
        chart_input = item.get("chart_data", {})
        try:
            # Step 1: Generate matplotlib code with LLM
            code_response = (PYTHON_CODE_PROMPT | chat_llm).invoke({
                "title": chart_input["title"],
                "chart_type": chart_input["chart_type"],
                "x_labels": chart_input["x_labels"],
                "y_values": chart_input["y_values"],
                "units": chart_input["units"]
            })

            # FIX: Extract content from AIMessage
            code_block = code_response.content.strip()

            # Remove markdown fences if present
            if code_block.startswith("```python"):
                code_block = code_block.replace("```python", "").strip()
            if code_block.endswith("```"):
                code_block = code_block[:-3].strip()

            # Step 2: Execute and save chart
            image_path = run_python_code_and_save_image(code_block)

            visualizations.append({
                "document_index": item["document_index"],
                "raw_snippet": item["raw_snippet"],
                "visualization_goal": item["visualization_goal"],
                "python_code": code_block,
                "chart_image_path": image_path
            })

        except Exception as e:
            print(f"[VisualizerAgent] Failed on doc #{item['document_index']}: {e}")
            traceback.print_exc()

    state["visualization_outputs"] = visualizations

    print(f"[VisualizerAgent] ✅ Rendered {len(visualizations)} charts to PNG")
    if visualizations:
        print("[VisualizerAgent] 🖼️ Sample Output:")
        print(json.dumps(visualizations[0], indent=2))

    return state
