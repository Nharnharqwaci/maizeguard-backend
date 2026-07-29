import io
import logging
import os
from typing import Dict, Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy",
    "MSV",
    "Northern_Leaf_Blight",
    "Southern_Leaf_Blight",
]

_session = None
_model_path = os.getenv("MODEL_PATH", "model.onnx")


def _get_session():
    global _session
    if _session is None:
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session_providers = [p for p in preferred if p in providers] or ["CPUExecutionProvider"]

            logger.info(
                f"[ONNX] Loading backend model from {_model_path}, providers={session_providers}"
            )
            _session = ort.InferenceSession(_model_path, providers=session_providers)
            logger.info("[ONNX] Backend model loaded successfully")
        except Exception as e:
            logger.error(f"[ONNX] Failed to load backend model: {e}")
            raise RuntimeError(
                f"Backend ONNX model failed to load: {e}. "
                f"Ensure model.onnx exists at backend root or set MODEL_PATH env var."
            )
    return _session


def run_inference(image_bytes: bytes) -> Dict[str, Any]:
    """
    Run backend ONNX inference.
    Input:  raw image bytes
    Output: {
        "class": str,
        "confidence": float (0-1 fraction),
        "all_probs": dict[str, float] (0-1 fractions)
    }
    """
    session = _get_session()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    # [1, 224, 224, 3] float32, pixels 0-255 (matches frontend)
    img_array = np.array(image, dtype=np.float32)
    input_data = np.expand_dims(img_array, axis=0)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    raw_probs = outputs[0][0]  # shape: [6]

    # Softmax guard: apply if outputs don't look like probabilities
    prob_sum = float(np.sum(raw_probs))
    if prob_sum < 0.9 or prob_sum > 1.1 or np.any(raw_probs < 0):
        exp_probs = np.exp(raw_probs - np.max(raw_probs))
        probs = exp_probs / np.sum(exp_probs)
    else:
        probs = raw_probs

    all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    max_idx = int(np.argmax(probs))

    return {
        "class": CLASS_NAMES[max_idx],
        "confidence": float(probs[max_idx]),
        "all_probs": all_probs,
    }