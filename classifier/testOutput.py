import joblib
#from sentence_transformers import SentenceTransformer
from scripts.llm.embed import embed_llm
import os
BASE_DIR = os.path.dirname(__file__)
# Load classifier and embedding model
clf = joblib.load(os.path.join(BASE_DIR, "classifier.joblib"))


# Example inputs
new_queries = [
    "Hi",
    "Who is the owner of the nats",
    "What is the current gas price"
]

# Generate embeddings
embeddings = embed_llm.embed_documents(new_queries)

# Predict
predictions = clf.predict(embeddings)
probabilities = clf.predict_proba(embeddings)

# Interpret results
for query, label, prob in zip(new_queries, predictions, probabilities):
    route = "RAG" if label == 1 else "GPT"
    confidence = max(prob)
    print(f"Query: {query}\nRoute: {route}, Confidence: {confidence:.2f}\n")
