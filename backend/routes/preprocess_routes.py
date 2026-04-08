from flask import Blueprint, request, jsonify
from services.preprocess import encode_sequence

preprocess_bp = Blueprint("preprocess", __name__)

@preprocess_bp.route("/preprocess", methods=["POST"])
def preprocess():
    data = request.json
    sequence = data.get("sequence")

    encoded = encode_sequence(sequence)

    return jsonify({
        "sequence": sequence,
        "encoded": encoded,
        "length": len(encoded)
    })