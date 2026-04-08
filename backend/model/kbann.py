import model.loader as loader
from services.preprocess import encode_sequence
from services.kbann_features import extract_kbann_features
import numpy as np

label_map_reverse = {
    0: "EI",
    1: "IE",
    2: "N"
}

def _build_probabilities(probabilities):
    return {
        label_map_reverse[idx]: float(probabilities[idx])
        for idx in range(len(probabilities))
    }


def get_kbann_prediction_details(sequence):
    if loader.kbann_model is None:
        raise RuntimeError("KBANN model is not loaded. Start the app after successful model loading.")

    encoded = encode_sequence(sequence)

    features = extract_kbann_features(sequence)
    feature_values = list(features.values())

    final_input = encoded + feature_values

    prediction_idx = int(loader.kbann_model.predict([final_input])[0])
    class_probabilities = loader.kbann_model.predict_proba([final_input])[0]
    confidence = float(np.max(class_probabilities))

    return {
        "prediction": label_map_reverse[prediction_idx],
        "confidence": confidence,
        "probabilities": _build_probabilities(class_probabilities),
        "features": features
    }


def predict_kbann(sequence):
    details = get_kbann_prediction_details(sequence)

    return details["prediction"], details["confidence"]
