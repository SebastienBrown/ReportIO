from flask import Flask, request, jsonify
import sys
import os
from flask_cors import CORS
import threading

# Make sure we can import from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.orchestrator import run_orchestration_pipeline

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])


# ---- Log buffer (global) ----
log_lines = []
last_result = None 

def log(msg):
    print(msg)  # still prints to console
    log_lines.append(msg)

@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Missing query"}), 400

    log_lines.clear()

    # Run your orchestration in a background thread
    def run():
        global last_result
        result = run_orchestration_pipeline(query, logger=log)
        last_result = result
        log("[✅] Final answer ready.")

    threading.Thread(target=run).start()

    return jsonify({"status": "started"}), 202

@app.route("/api/last_result", methods=["GET"])
def get_last_result():
    return jsonify(last_result or {})

@app.route("/api/logs", methods=["GET"])
def get_logs():
    return jsonify({"logs": log_lines})


if __name__ == "__main__":
    app.run(debug=True)
