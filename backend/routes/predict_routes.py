from flask import Blueprint, request, jsonify
from model.kbann import get_kbann_prediction_details
from model.deep_model import get_dl_prediction_details

predict_bp = Blueprint("predict", __name__)


def _validate_sequence(payload):
    if not payload:
        return None
    sequence = payload.get("sequence")
    if not sequence or not sequence.strip():
        return None
    return sequence.strip()


@predict_bp.route("/predict/kbann", methods=["POST"])
def predict():
    sequence = _validate_sequence(request.json)
    if sequence is None:
        return jsonify({"error": "A non-empty 'sequence' field is required."}), 400

    details = get_kbann_prediction_details(sequence)

    return jsonify({
        "model": "KBANN",
        "sequence": sequence,
        "prediction": details["prediction"],
        "confidence": details["confidence"],
        "probabilities": details["probabilities"],
        "features": details["features"]
    })


@predict_bp.route("/predict/dl", methods=["POST"])
def predict_dl_route():
    sequence = _validate_sequence(request.json)
    if sequence is None:
        return jsonify({"error": "A non-empty 'sequence' field is required."}), 400

    details = get_dl_prediction_details(sequence)

    return jsonify({
        "model": "Deep Learning",
        "sequence": sequence,
        "prediction": details["prediction"],
        "confidence": details["confidence"],
        "probabilities": details["probabilities"]
    })


@predict_bp.route("/predict/compare", methods=["POST"])
def predict_compare_route():
    sequence = _validate_sequence(request.json)
    if sequence is None:
        return jsonify({"error": "A non-empty 'sequence' field is required."}), 400

    kbann_result = get_kbann_prediction_details(sequence)
    dl_result = get_dl_prediction_details(sequence)

    return jsonify({
        "sequence": sequence,
        "KBANN": kbann_result,
        "DeepLearning": dl_result
    })
