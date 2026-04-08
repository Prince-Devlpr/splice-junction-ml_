import os
import json
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from services.data_service import load_data
from services.preprocess import encode_sequence


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

    encoded = encode_sequence(seq)

    # reshape into (sequence_length, 4)
    encoded = np.array(encoded).reshape(-1, 4)

    X.append(encoded)
    y.append(label_map[row["label"]])


X = np.array(X)
y = np.array(y)


# 🔹 One-hot labels
y_cat = to_categorical(y, num_classes=3)


# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)


# 🧠 MODEL
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(X.shape[1], 4)),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=64, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# 🔹 Train
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# 🔹 Evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred)


print("\n===== DL MODEL METRICS =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)


# 🔹 Save model
model.save("models/dl_model.h5")
print("\nDL model saved → models/dl_model.h5")


# 🔹 Save metrics
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist()
}

with open("models/dl_metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved → models/dl_metrics.json")