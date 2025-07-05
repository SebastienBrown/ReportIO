import json

# Load queries from jsonl file (one JSON object per line)
with open("queries.json", "r", encoding="utf-8") as f:
    queries_dict = json.load(f)
queries = queries_dict["queries"]  # Extract the list of query strings


with open("labels.json", "r") as f:
    labels_dict = json.load(f)
labels = labels_dict["class"]  # extract the list of labels

print(len(queries))
print(len(labels))
assert len(queries) == len(labels), "Queries and labels count mismatch"


from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 1. Load SentenceTransformer (lightweight model recommended)
model = SentenceTransformer('all-MiniLM-L6-v2')  # ~22MB, very fast
print("model loaded")

# 3. Encode Queries
embeddings = model.encode(queries)
print("embeddings encoded")

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# 5. Train Classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("fit done")

# 6. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Save Model
joblib.dump(clf, "classifier.joblib")
joblib.dump(model, "sentence_transformer.joblib")
