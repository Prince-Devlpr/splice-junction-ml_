def extract_kbann_features(seq):
    seq = seq.upper()

    mid = len(seq) // 2

    # window around center
    window = seq[mid-3:mid+3]

    features = {
        "has_GT_center": 1 if "GT" in window else 0,
        "has_AG_center": 1 if "AG" in window else 0,
        "count_G": seq.count("G"),
        "count_A": seq.count("A"),
        "count_T": seq.count("T"),
        "count_C": seq.count("C"),
    }

    return features