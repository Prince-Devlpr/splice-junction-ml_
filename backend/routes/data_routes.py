from flask import Blueprint, jsonify
from services.data_service import get_sample_data

data_bp = Blueprint("data", __name__)

@data_bp.route("/data", methods=["GET"])
def get_data():
    df = get_sample_data()
    return jsonify(df.to_dict(orient="records"))