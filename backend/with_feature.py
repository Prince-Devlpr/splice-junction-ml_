import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# =========================
# Load data
# =========================
data = pd.read_csv("splice.data", header=None)
data[2] = data[2].astype(str).str.replace(" ", "").str.upper()
sequences = data[2]

# =========================
# 1) Class Distribution
# =========================
data[0].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =========================
# 2) Nucleotide Frequency
# =========================
all_seq = "".join(sequences)
counts = Counter(all_seq)

nucleotides = ["A", "C", "G", "T"]
values = [counts[n] for n in nucleotides]

plt.bar(nucleotides, values)
plt.title("Nucleotide Frequency")
plt.xlabel("Nucleotide")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =========================
# 3) Position-wise Distribution Heatmap
# =========================
mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
seq_len = len(sequences.iloc[0])
matrix = np.zeros((seq_len, 4))

for seq in sequences:
    for i, char in enumerate(seq):
        if char in mapping:
            matrix[i][mapping[char]] += 1

plt.imshow(matrix.T, aspect="auto")
plt.title("Position-wise Nucleotide Distribution")
plt.xlabel("Position")
plt.ylabel("Nucleotide Index (A,C,G,T)")
plt.colorbar()
plt.tight_layout()
plt.show()

# =========================
# 4) K-mer features
# =========================
vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 2))
X_kmer = vectorizer.fit_transform(sequences).toarray()

# =========================
# 5) PCA Visualization
# =========================
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_kmer)

plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=data[0].astype("category").cat.codes,
    alpha=0.8,
)
plt.title("PCA Visualization After Preprocessing")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# =========================
# 6) Biological features
# =========================
def biological_features(seq):
    gt = 1 if "GT" in seq else 0
    ag = 1 if "AG" in seq else 0
    gc_content = (seq.count("G") + seq.count("C")) / len(seq)
    a_count = seq.count("A")
    c_count = seq.count("C")
    g_count = seq.count("G")
    t_count = seq.count("T")
    return [gt, ag, gc_content, a_count, c_count, g_count, t_count]

bio = np.array([biological_features(s) for s in sequences])
X = np.hstack((X_kmer, bio))

# =========================
# 7) Encode labels and train
# =========================
le = LabelEncoder()
y = le.fit_transform(data[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("KBANN WITH Feature Extraction Accuracy:", accuracy)

# =========================
# 8) Accuracy graph
# =========================
plt.bar(["KBANN With Feature"], [accuracy], color="seagreen")
plt.title("KBANN Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
