import numpy as np
import model.loader as loader
from services.preprocess import encode_sequence

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


def get_dl_prediction_details(sequence):
    if loader.dl_model is None:
        raise RuntimeError("Deep Learning model is not loaded. Start the app after successful model loading.")

    encoded = encode_sequence(sequence)

    # reshape for CNN
    encoded = np.array(encoded).reshape(-1, 4)
    encoded = encoded.reshape(1, encoded.shape[0], 4)

    pred_probs = loader.dl_model.predict(encoded)
    probabilities = pred_probs[0]
    prediction_idx = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return {
        "prediction": label_map_reverse[prediction_idx],
        "confidence": confidence,
        "probabilities": _build_probabilities(probabilities)
    }


def predict_dl(sequence):
    details = get_dl_prediction_details(sequence)

    return details["prediction"], details["confidence"]
