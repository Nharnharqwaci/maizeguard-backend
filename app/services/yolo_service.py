import numpy as np
import logging
import os
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_PATH = Path(__file__).parent.parent / "models" / "best.keras"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Keras model not found at {MODEL_PATH}. "
        "Place your trained best.keras inside app/models/"
    )

# must match train_gen.class_indices order — alphabetical
CLASS_NAMES = [
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy",
    "MSV",
    "Northern_Leaf_Blight",
    "Southern_Leaf_Blight",
]

# all 6 are valid — anything outside these returns Uncertain
VALID_CLASSES = set(CLASS_NAMES)
CONFIDENCE_THRESHOLD = 0.70

IMG_HEIGHT = 224
IMG_WIDTH  = 224

try:
    import tensorflow as tf
    model = tf.keras.models.load_model(str(MODEL_PATH))
    logger.info(f"Keras model loaded from {MODEL_PATH}")
    logger.info(f"Input shape:  {model.input_shape}")
    logger.info(f"Output shape: {model.output_shape}")
    logger.info(f"Classes: {CLASS_NAMES}")
except Exception as e:
    raise RuntimeError(f"Failed to load Keras model: {e}")


def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(str(image_path)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32) 
    return np.expand_dims(arr, axis=0)


def run_inference(image_path: str) -> dict:
    try:
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array, verbose=0)
        probs = predictions[0]

        class_id   = int(np.argmax(probs))
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

        # low confidence means image is likely not a maize leaf
        # or does not clearly belong to any class
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