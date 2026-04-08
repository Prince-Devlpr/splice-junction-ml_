from flask import Flask
from flask_cors import CORS

from routes.data_routes import data_bp
from routes.analytics_routes import analytics_bp
from routes.predict_routes import predict_bp
from routes.preprocess_routes import preprocess_bp
from routes.preprocess_data_routes import preprocess_data_bp
from routes.kbann_routes import kbann_bp
from routes.metrics_routes import metrics_bp
from model.loader import load_models

load_models()

app = Flask(__name__)
CORS(app)

app.register_blueprint(data_bp)
app.register_blueprint(analytics_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(preprocess_bp)
app.register_blueprint(preprocess_data_bp)
app.register_blueprint(kbann_bp)
app.register_blueprint(metrics_bp)
@app.route("/")
def home():
    return "Backend Running"

if __name__ == "__main__":
    app.run(debug=True)
    