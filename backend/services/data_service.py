import pandas as pd
import numpy as np
from config import DATA_PATH
from services.preprocess import encode_sequence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
def load_data():
    df = pd.read_csv("splice.data", header=None)

    df.columns = ["label", "id", "sequence"]

    # CLEAN sequence
    df["sequence"] = df["sequence"].str.replace(" ", "")
    df["sequence"] = df["sequence"].str.strip()

    return df

def get_sample_data(n=20):
    df = load_data()
    return df.head(n)

def get_class_distribution():
    df = load_data()
    return df["label"].value_counts().to_dict()



def get_preprocessed_sample(n=10):
    df = load_data().head(n)

    processed = []

    for _, row in df.iterrows():
        encoded = encode_sequence(row["sequence"])

        processed.append({
            "label": row["label"],
            "sequence": row["sequence"],
            "encoded": encoded[:20]  # show only first 20 values (important!)
        })

    return processed

def get_sequence_lengths():
    df = load_data()
    lengths = df["sequence"].apply(len)
    return lengths.tolist()

def get_nucleotide_frequency():
    df = load_data()

    counts = {"A":0, "T":0, "G":0, "C":0}

    for seq in df["sequence"]:
        for char in seq:
            if char in counts:
                counts[char] += 1

    return counts


def get_positionwise_distribution():
    df = load_data()
    sequences = df["sequence"].astype(str).str.upper().tolist()
    if not sequences:
        return []

    seq_len = min(len(seq) for seq in sequences)
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    matrix = np.zeros((seq_len, 4), dtype=int)

    for seq in sequences:
        for i, char in enumerate(seq[:seq_len]):
            if char in mapping:
                matrix[i][mapping[char]] += 1

    rows = []
    for i in range(seq_len):
        rows.append({
            "position": i + 1,
            "A": int(matrix[i][0]),
            "C": int(matrix[i][1]),
            "G": int(matrix[i][2]),
            "T": int(matrix[i][3])
        })

    return rows


def get_pca_projection(limit=500):
    df = load_data()
    sequences = df["sequence"].astype(str).str.upper().tolist()
    labels = df["label"].tolist()
    if not sequences:
        return {"points": [], "explained_variance": [0.0, 0.0]}

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 2))
    x_kmer = vectorizer.fit_transform(sequences).toarray()

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(x_kmer)

    # Build a balanced sample across labels so EI/IE/N are all visible in the chart.
    label_to_indices = {}
    for i, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)

    total_available = len(sequences)
    capped = min(limit, total_available)
    unique_labels = list(label_to_indices.keys())
    per_label_quota = max(1, capped // max(1, len(unique_labels)))

    selected_indices = []
    for label in unique_labels:
        selected_indices.extend(label_to_indices[label][:per_label_quota])

    if len(selected_indices) < capped:
        used = set(selected_indices)
        for i in range(total_available):
            if i not in used:
                selected_indices.append(i)
                if len(selected_indices) == capped:
                    break

    points = []
    for i in selected_indices:
        points.append({
            "pc1": float(reduced[i, 0]),
            "pc2": float(reduced[i, 1]),
            "label": labels[i]
        })

    return {
        "points": points,
        "explained_variance": [
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1])
        ]
    }
