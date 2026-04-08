from flask import Blueprint, jsonify
from services.data_service import get_preprocessed_sample

preprocess_data_bp = Blueprint("preprocess_data", __name__)

@preprocess_data_bp.route("/data/preprocessed", methods=["GET"])
def preprocessed_data():
    data = get_preprocessed_sample()
    return jsonify(data)