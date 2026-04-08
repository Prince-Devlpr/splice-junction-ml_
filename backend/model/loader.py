import pickle
import json
from tensorflow.keras.models import load_model

kbann_model = None
dl_model = None


def load_models():
    global kbann_model, dl_model

    # KBANN
    try:
        with open("models/kbann_model.pkl", "rb") as f:
            kbann_model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(
            "Failed to load models/kbann_model.pkl. "
            "This usually means numpy/scikit-learn versions differ from training. "
            "Use requirements.txt pinned versions or retrain and re-export the model."
        ) from e

    # DL
    dl_model = load_model("models/dl_model.h5")
