from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from datetime import datetime
import logging
from app.services.image_service import save_image
from app.services.yolo_service import run_inference
from app.core.database import scans_collection

logger = logging.getLogger(__name__)
router = APIRouter()


def get_treatment(prediction: str) -> list[str]:
    treatments = {
        "Healthy": [
            "No disease detected — your maize plant looks healthy.",
            "Continue regular crop monitoring every 7 days.",
            "Maintain proper NPK fertilization (60-40-40 kg/ha baseline).",
            "Ensure adequate spacing between plants for good airflow.",
            "Keep field free of weeds that may host pests or disease.",
        ],
        "MSV": [
            "Maize Streak Virus (MSV) detected — act immediately.",
            "Remove and destroy severely infected plants to prevent spread.",
            "Control leafhoppers (Cicadulina spp.) using imidacloprid or thiamethoxam insecticides.",
            "Plant MSV-resistant varieties in the next season (e.g. SAMMAZ series).",
            "Avoid replanting maize immediately after an infected crop in the same field.",
            "Apply reflective mulches to deter leafhopper populations.",
            "Report outbreak to your local agricultural extension officer.",
        ],
        "MLS": [
            "Maize Lethal Senescence (MLS) detected — severe disease, act urgently.",
            "There is no cure — uproot and burn all infected plants immediately.",
            "Do not compost infected plant material as it spreads the pathogen.",
            "Disinfect tools used on infected plants with 70% alcohol or bleach solution.",
            "Control insect vectors (planthoppers and leafhoppers) with insecticides.",
            "Plant certified disease-free seed in the next season.",
            "Quarantine affected field sections and notify your agricultural extension officer.",
            "Rotate crops with non-host species such as legumes or cowpea for at least one season.",
        ],
        "Not_Maize": [
            "This image does not appear to be a maize leaf.",
            "Please upload a clear close-up photo of a maize leaf in natural daylight.",
            "Avoid blurry images or photos of soil, hands, or other plants.",
            "Ensure the leaf fills most of the frame and is in sharp focus.",
        ],
        "Uncertain": [
            "The image could not be confidently classified.",
            "Please upload a clearer, well-lit maize leaf image.",
            "Ensure the leaf fills most of the frame and is in focus.",
            "Try photographing in natural daylight without flash.",
        ],
    }
    return treatments.get(prediction, treatments["Uncertain"])


def get_severity(prediction: str) -> str:
    return {
        "Healthy":   "none",
        "MSV":       "high",
        "MLS":       "critical",
        "Not_Maize": "none",
        "Uncertain": "low",
    }.get(prediction, "low")


def get_color(prediction: str) -> str:
    return {
        "Healthy":   "green",
        "MSV":       "orange",
        "MLS":       "red",
        "Not_Maize": "gray",
        "Uncertain": "yellow",
    }.get(prediction, "gray")


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: str = Form(None)
):
    try:
        # validate file type
        if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a JPG, PNG, or WEBP image."
            )

        # save uploaded image
        filepath = save_image(file)

        # run YOLO inference
        result = run_inference(filepath)

        raw_prediction = result["class"]
        confidence_score = result["confidence"]
        confidence_pct = round(confidence_score * 100, 2)
        all_probs = result.get("all_probs", {})

        # normalize class name
        class_map = {
            "healthy":   "Healthy",
            "msv":       "MSV",
            "mls":       "MLS",
            "mlb":        "MLS",
            "maize lethal senescence":  "MLS",
            "not_maize": "Not_Maize",
            "not maize": "Not_Maize",
            "not-maize": "Not_Maize",
        }
        prediction = class_map.get(
            raw_prediction.lower().strip(),
            raw_prediction
        )

        # get treatment and metadata
        treatment = get_treatment(prediction)
        severity = get_severity(prediction)
        color = get_color(prediction)

        # save scan to MongoDB
        scan_doc = {
            "user_id": user_id,
            "image_url": str(filepath),
            "prediction": prediction,
            "confidence": confidence_pct,
            "severity": severity,
            "treatment": treatment,
            "all_probs": {k: round(v * 100, 2) for k, v in all_probs.items()},
            "created_at": datetime.utcnow()
        }

        insert_result = await scans_collection.insert_one(scan_doc)
        logger.info(f"Scan saved — id: {insert_result.inserted_id}, prediction: {prediction}")
        

        return {
            "id": str(insert_result.inserted_id),
            "prediction": prediction,
            "confidence": confidence_pct,
            "severity": severity,
            "color": color,
            "treatment": treatment,
            "all_probs": {k: round(v * 100, 2) for k, v in all_probs.items()},
        }
        

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

