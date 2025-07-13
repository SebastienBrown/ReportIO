import sys
import os



from scripts.pipelines.text_pipeline.orchestrator import run_orchestration_pipeline as run_text_pipeline
# from pipelines.video_pipeline.orchestrator import run_video_pipeline  # Uncomment when ready


def run_multimodal_pipeline(query: str, logger=print):
    logger("[DEBUG] [Multimodal] Starting text pipeline...")
    text_result = run_text_pipeline(query, logger=logger)

    # Stubbed for now – uncomment once video pipeline is ready
    # logger("[DEBUG] [Multimodal] Running video pipeline...")
    # video_result = run_video_pipeline(query, logger=logger)

    logger("[DEBUG] [Multimodal] Combined results ready.")

    return {
        "status": "ok",
        "llm_answer": text_result.get("llm_answer"),
        "top_snippets": text_result.get("top_snippets", []),
        "retrieved_chunks": text_result.get("retrieved_chunks", []),
        # "videos": video_result.get("top_videos", [])  # future addition
    }
