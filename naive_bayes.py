import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# =========================
# Load dataset
# =========================
data = pd.read_csv("data/splice.data", header=None)

# =========================
# Clean DNA sequences
# =========================
data[2] = data[2].astype(str).str.replace(" ", "").str.upper()

sequences = data[2]

# =========================
# Feature extraction (k-mer)
# =========================
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,2))

X = vectorizer.fit_transform(sequences).toarray()

print("Feature shape:", X.shape)

# =========================
# Encode labels
# =========================
le = LabelEncoder()

y = le.fit_transform(data[0])

print("Classes:", le.classes_)

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Train Naive Bayes model
# =========================
nb = GaussianNB()

nb.fit(X_train, y_train)

# =========================
# Predictions
# =========================
y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nNaive Bayes Accuracy:", accuracy)