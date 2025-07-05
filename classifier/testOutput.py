import joblib
from sentence_transformers import SentenceTransformer

# Load classifier and embedding model
clf = joblib.load("classifier.joblib")
embedder = SentenceTransformer('all-MiniLM-L6-v2') 

# Example inputs
new_queries = [
    "Hi",
    "Who is the owner of the nats",
    "What is the current gas price"
]

# Generate embeddings
embeddings = embedder.encode(new_queries)

# Predict
predictions = clf.predict(embeddings)
probabilities = clf.predict_proba(embeddings)

# Interpret results
for query, label, prob in zip(new_queries, predictions, probabilities):
    route = "RAG" if label == 1 else "GPT"
    confidence = max(prob)
    print(f"Query: {query}\nRoute: {route}, Confidence: {confidence:.2f}\n")
