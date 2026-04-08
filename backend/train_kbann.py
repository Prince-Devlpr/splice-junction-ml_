import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from services.data_service import load_data
from services.preprocess import encode_sequence
from services.kbann_features import extract_kbann_features


# 🔹 Ensure models folder exists
os.makedirs("models", exist_ok=True)


# 🔹 Load dataset
df = load_data()

X = []
y = []

label_map = {
    "EI": 0,
    "IE": 1,
    "N": 2
}


# 🔹 Prepare data
for _, row in df.iterrows():
    seq = row["sequence"]

    # Encode sequence
    encoded = encode_sequence(seq)

    # Extract rule-based features
    features = extract_kbann_features(seq)
    feature_values = list(features.values())

    # Combine both
    final_input = encoded + feature_values

    X.append(final_input)
    y.append(label_map[row["label"]])


X = np.array(X)
y = np.array(y)


# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 🔹 Model
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=20,
    verbose=True
)


# 🔹 Train
model.fit(X_train, y_train)


# 🔹 Predict
y_pred = model.predict(X_test)


# 🔹 Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred)


# 🔹 Print results
print("\n===== KBANN MODEL METRICS =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)


# 🔹 Save model
with open("models/kbann_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved → models/kbann_model.pkl")


# 🔹 Save metrics
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist()
}

with open("models/kbann_metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved → models/kbann_metrics.json")