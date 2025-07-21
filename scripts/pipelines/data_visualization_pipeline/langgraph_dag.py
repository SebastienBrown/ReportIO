from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

from scripts.pipelines.data_visualization_pipeline.agents.query_rephraser import query_rephraser
from scripts.pipelines.data_visualization_pipeline.agents.search_and_scrape import search_and_scrape
from scripts.pipelines.data_visualization_pipeline.agents.analyzer_agent import analyzer_agent
from scripts.pipelines.data_visualization_pipeline.agents.extractor_agent import extractor_agent
from scripts.pipelines.data_visualization_pipeline.agents.visualizer_agent import visualizer_agent

# Dummy formatter if not defined yet
def formatter(state: dict) -> dict:
    print("[Formatter] 🧾 Returning final state...")
    return state


# Build the DAG
builder = StateGraph(dict)

# Add nodes
builder.add_node("rephraser", query_rephraser)
builder.add_node("search_and_scrape", search_and_scrape)
builder.add_node("analyzer_agent", analyzer_agent)
builder.add_node("extractor_agent", extractor_agent)
builder.add_node("visualizer", RunnableLambda(visualizer_agent))  # FIXED
builder.add_node("formatter", RunnableLambda(formatter))

# Wire up edges
builder.set_entry_point("rephraser")
builder.add_edge("rephraser", "search_and_scrape")
builder.add_edge("search_and_scrape", "analyzer_agent")
builder.add_edge("analyzer_agent", "extractor_agent")
builder.add_edge("extractor_agent", "visualizer")
builder.add_edge("visualizer", "formatter")
builder.set_finish_point("formatter")

# Compile DAG
graph = builder.compile()


# Run test
if __name__ == "__main__":
    user_query = "What is the weather in NYC over the next few days?"
    result = graph.invoke({"user_query": user_query})
    print("\n--- Final Output ---")
    #print(result)
