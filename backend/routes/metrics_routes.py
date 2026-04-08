from flask import Blueprint, jsonify
import json

metrics_bp = Blueprint("metrics", __name__)

@metrics_bp.route("/metrics/kbann", methods=["GET"])
def get_metrics():
    with open("models/kbann_metrics.json") as f:
        data = json.load(f)
    return jsonify(data)

@metrics_bp.route("/metrics/dl", methods=["GET"])
def get_dl_metrics():
    with open("models/dl_metrics.json") as f:
        data = json.load(f)
    return jsonify(data)

@metrics_bp.route("/metrics/compare", methods=["GET"])
def compare_models():
    with open("models/kbann_metrics.json") as f:
        kbann = json.load(f)

    with open("models/dl_metrics.json") as f:
        dl = json.load(f)

    return jsonify({
        "KBANN": kbann,
        "DeepLearning": dl
    })