import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# =========================
# Load dataset
# =========================
data = pd.read_csv("data/splice.data", header=None)

# =========================
# Clean sequences
# =========================
data[2] = data[2].astype(str).str.replace(" ", "").str.upper()

sequences = data[2]

# =========================
# Feature extraction (2-mer)
# =========================
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,2))

X = vectorizer.fit_transform(sequences).toarray()

# =========================
# Encode labels
# =========================
le = LabelEncoder()
y = le.fit_transform(data[0])

print("Classes:", le.classes_)

# =========================
# Train test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Train Random Forest
# =========================
rf = RandomForestClassifier(n_estimators=200, random_state=42)

rf.fit(X_train, y_train)

# =========================
# Model Accuracy
# =========================
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Accuracy:", accuracy)


