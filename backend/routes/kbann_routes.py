from flask import Blueprint, request, jsonify
from services.kbann_features import extract_kbann_features

kbann_bp = Blueprint("kbann", __name__)

@kbann_bp.route("/kbann/features", methods=["POST"])
def kbann_features():
    data = request.json
    seq = data.get("sequence")

    features = extract_kbann_features(seq)

    return jsonify({
        "sequence": seq,
        "features": features
    })