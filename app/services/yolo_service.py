from ultralytics import YOLO
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found at {MODEL_PATH}. "
        "Place your trained best.pt inside app/models/"
    )

try:
    model = YOLO(str(MODEL_PATH))
    logger.info(f"YOLO model loaded — classes: {model.names}")
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

CONFIDENCE_THRESHOLD = 0.70


def run_inference(image_path: str) -> dict:
    try:
        results = model(str(image_path), verbose=False)

        if not results or results[0].probs is None:
            logger.warning(f"No inference results for: {image_path}")
            return {"class": "Uncertain", "confidence": 0.0, "all_probs": {}}

        probs = results[0].probs
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        class_name = model.names[class_id]

        all_probs = {
            model.names[i]: round(float(probs.data[i]), 4)
            for i in range(len(model.names))
        }

        logger.info(
            f"Inference — class: {class_name}, "
            f"confidence: {confidence:.2%}, probs: {all_probs}"
        )

        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                f"Low confidence ({confidence:.2%}) — returning Uncertain"
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