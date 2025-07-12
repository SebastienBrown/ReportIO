# scripts/query_classifier.py

import os
import joblib
from scripts.llm.embed import embed_llm

# Load the classifier once at module level
BASE_DIR = os.path.dirname(__file__)
clf = joblib.load(os.path.join(BASE_DIR, "../classifier/classifier.joblib"))

def classify_query(query: str):
    """Classify a query as 'RAG' or 'GPT' using the pretrained classifier."""
    embedding = embed_llm.embed_query(query)
    label = clf.predict([embedding])[0]
    prob = max(clf.predict_proba([embedding])[0])
    route = "RAG" if label == 1 else "GPT"
    return {
        "query": query,
        "route": route,
        "confidence": prob
    }
