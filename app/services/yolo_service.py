import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Suppress TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_PATH = Path(__file__).parent.parent / "models" / "best.keras"

CLASS_NAMES = [
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy",
    "MSV",
    "Northern_Leaf_Blight",
    "Southern_Leaf_Blight",
]

VALID_CLASSES = set(CLASS_NAMES)
CONFIDENCE_THRESHOLD = 0.70
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Global singleton — stays None until first inference call
_model = None


def get_model():
    """Lazy-load the Keras model. Loads once, then cached in memory."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Keras model not found at {MODEL_PATH}. "
                "Place your trained best.keras inside app/models/"
            )

        try:
            import tensorflow as tf
            logger.info(f"Loading Keras model from {MODEL_PATH}...")
            _model = tf.keras.models.load_model(str(MODEL_PATH))
            logger.info(f"Model loaded. Input: {_model.input_shape}, Output: {_model.output_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")

    return _model


def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(str(image_path)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def run_inference(image_path: str) -> dict:
    try:
        model = get_model()  # Lazy load happens here on first call
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array, verbose=0)
        probs = predictions[0]

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        class_name = CLASS_NAMES[class_id]

        all_probs = {
            CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(CLASS_NAMES))
        }

        logger.info(
            f"Inference — class: '{class_name}', "
            f"confidence: {confidence:.2%}, "
            f"all_probs: {all_probs}"
        )

        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                f"Confidence {confidence:.2%} below threshold "
                f"{CONFIDENCE_THRESHOLD:.0%} — returning Uncertain"
            )
            return {
                "class": "Uncertain",
                "confidence": confidence,
                "all_probs": all_probs
            }

        return {
            "class": class_name,
            "confidence": confidence,
            "all_probs": all_probs
        }

    except FileNotFoundError:
        logger.error(f"Image not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise RuntimeError(f"Inference failed: {e}")