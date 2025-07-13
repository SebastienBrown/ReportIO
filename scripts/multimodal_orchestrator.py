from scripts.pipelines.text_pipeline.orchestrator import run_orchestration_pipeline as run_text_pipeline
from scripts.pipelines.video_pipeline.orchestrator import run_video_pipeline

def run_multimodal_pipeline(query: str, logger=print):
    logger("[DEBUG] [Multimodal] Starting text pipeline...")
    text_result = run_text_pipeline(query, logger=lambda msg: logger(f"[Text] {msg}"))

    logger("[DEBUG] [Multimodal] Starting video pipeline...")
    video_result = run_video_pipeline(query, logger=lambda msg: logger(f"[Video] {msg}"))

    logger("[DEBUG] [Multimodal] Combined results ready.")

    return {
        "status": "ok",
        "llm_answer": text_result.get("llm_answer"),
        "top_snippets": text_result.get("top_snippets", []),
        "retrieved_chunks": text_result.get("retrieved_chunks", []),
        "videos": video_result.get("retrieved_chunks", [])  # or `video_result["retrieved_chunks"]`
    }
