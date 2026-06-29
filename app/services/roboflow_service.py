# app/services/yolo_service.py

from ultralytics import YOLO

model = YOLO(
    "app/models/best.pt"
)


def run_inference(image_path):

    results = model.predict(
        source=image_path,
        verbose=False
    )

    probs = results[0].probs

    class_id = probs.top1

    confidence = float(
        probs.top1conf
    )

    prediction = (
        model.names[class_id]
    )

    return {
        "prediction": prediction,
        "confidence": round(
            confidence * 100,
            2
        )
    }