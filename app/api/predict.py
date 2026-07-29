from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Header
from datetime import datetime
import logging
import math
import asyncio
from typing import List, Dict

from app.services.image_service import save_image
from app.services.youtube_service import get_videos
from app.services.onnx_inference_service import run_inference as onnx_run_inference
from app.core.database import scans_collection
from app.core.security import decode_access_token

logger = logging.getLogger(__name__)
router = APIRouter()

CONFIDENCE_THRESHOLD = 70.0
ENTROPY_THRESHOLD = 0.5


# ── HELPERS ──


def compute_normalized_entropy(probs: dict[str, float]) -> float:
    """
    Robust entropy calculator.
    Accepts fractions (0-1) OR percentages (0-100) automatically.
    """
    values = [p for p in probs.values() if p > 0]
    if len(values) <= 1:
        return 0.0
    # Auto-detect percentages and normalize
    if any(v > 1 for v in values):
        values = [v / 100 for v in values]
    entropy = -sum(p * math.log(p) for p in values)
    max_entropy = math.log(len(values))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def get_treatment(prediction: str, lang: str = "en") -> list[str]:
    treatments = {
        "Common_Rust": [
            "Common Rust (Puccinia sorghi) detected — fungal disease requiring prompt action.",
            "Apply fungicides containing azoxystrobin, propiconazole, or mancozeb.",
            "Remove heavily infected leaves.",
            "Plant rust-resistant maize varieties.",
            "Rotate maize with legumes.",
            "Maintain proper spacing for airflow.",
            "Monitor fields every week.",
        ],
        "Gray_Leaf_Spot": [
            "Gray Leaf Spot detected.",
            "Apply recommended foliar fungicides.",
            "Improve drainage.",
            "Practice crop rotation.",
            "Destroy infected crop residue.",
            "Use resistant maize varieties.",
        ],
        "Healthy": [
            "Excellent! No disease detected.",
            "Continue regular crop monitoring.",
            "Maintain proper fertilization.",
            "Keep weeds under control.",
            "Inspect plants weekly.",
            "Keep following good agronomic practices.",
        ],
        "MSV": [
            "Maize Streak Virus detected.",
            "Remove severely infected plants immediately.",
            "Control leafhopper vectors.",
            "Plant resistant maize varieties.",
            "Avoid continuous maize cropping.",
            "There is no cure for infected plants.",
            "Prevent spread to healthy plants.",
        ],
        "Northern_Leaf_Blight": [
            "Northern Leaf Blight detected.",
            "Apply fungicides early.",
            "Improve field airflow.",
            "Destroy infected residues.",
            "Rotate crops.",
            "Plant resistant hybrids.",
        ],
        "Southern_Leaf_Blight": [
            "Southern Leaf Blight detected.",
            "Apply fungicides immediately.",
            "Remove infected plants.",
            "Improve drainage.",
            "Use certified seed.",
            "Rotate crops.",
        ],
        "Uncertain": [
            "The image could not be confidently classified.",
            "Please upload a clearer maize leaf image.",
            "Ensure the leaf occupies most of the picture.",
            "Take the picture in natural daylight.",
            "Avoid blurry images.",
            "Make sure the image actually shows a maize leaf.",
        ],
    }
    return treatments.get(prediction, treatments["Uncertain"])


def get_severity(prediction: str) -> str:
    return {
        "Healthy": "none",
        "Common_Rust": "medium",
        "Gray_Leaf_Spot": "medium",
        "MSV": "high",
        "Northern_Leaf_Blight": "high",
        "Southern_Leaf_Blight": "critical",
        "Uncertain": "low",
    }.get(prediction, "low")


def get_color(prediction: str) -> str:
    return {
        "Healthy": "green",
        "Common_Rust": "orange",
        "Gray_Leaf_Spot": "purple",
        "MSV": "red",
        "Northern_Leaf_Blight": "amber",
        "Southern_Leaf_Blight": "rose",
        "Uncertain": "yellow",
    }.get(prediction, "yellow")


async def safe_get_videos(
    prediction: str, lang: str = "en", timeout: float = 8.0
) -> List[Dict]:
    try:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, get_videos, prediction, lang),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[YouTube] Timeout after {timeout}s for {prediction}")
        return []
    except Exception as e:
        logger.error(f"[YouTube] Unexpected error: {e}")
        return []


# ── ENDPOINTS ──


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    user_id: str = Form(None),
):
    """
    FALLBACK inference endpoint.
    Runs when frontend ONNX (browser) fails to load or crashes.
    Returns the exact same shape the frontend expects.
    """
    resolved_user_id = None
    if authorization:
        try:
            token = authorization.replace("Bearer ", "")
            payload = decode_access_token(token)
            resolved_user_id = payload.get("sub")
        except Exception as e:
            logger.warning(f"Failed to decode token: {e}")

    final_user_id = resolved_user_id or user_id

    try:
        if file.content_type not in [
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp",
        ]:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, PNG and WEBP images are supported.",
            )

        # 1. Save image (also validates it)
        filepath = save_image(file)

        # 2. Backend ONNX inference
        with open(filepath, "rb") as f:
            result = onnx_run_inference(f.read())

        prediction = result["class"]
        confidence = round(result["confidence"] * 100, 2)  # → percentage
        all_probs = {
            k: round(v * 100, 2) for k, v in result["all_probs"].items()
        }  # → percentages

        # 3. Threshold / uncertainty guard
        valid_classes = [
            "Common_Rust",
            "Gray_Leaf_Spot",
            "Healthy",
            "MSV",
            "Northern_Leaf_Blight",
            "Southern_Leaf_Blight",
        ]

        if prediction not in valid_classes:
            prediction = "Uncertain"
            logger.info(
                f"Rejected — unrecognized class from backend: {result.get('class')}"
            )
        else:
            norm_entropy = compute_normalized_entropy(all_probs)
            if confidence < CONFIDENCE_THRESHOLD:
                logger.info(
                    f"Rejected — confidence {confidence}% below threshold {CONFIDENCE_THRESHOLD}%"
                )
                prediction = "Uncertain"
            elif norm_entropy > ENTROPY_THRESHOLD:
                logger.info(
                    f"Rejected — entropy {norm_entropy:.2f} above threshold {ENTROPY_THRESHOLD}"
                )
                prediction = "Uncertain"

        # 4. Enrich response
        treatment = get_treatment(prediction)
        severity = get_severity(prediction)
        color = get_color(prediction)
        videos = await safe_get_videos(prediction)

        # 5. Best-effort DB save
        scan_id = None
        db_attempted = False
        try:
            if final_user_id:
                db_attempted = True
                scan_doc = {
                    "user_id": final_user_id,
                    "image_url": str(filepath),
                    "prediction": prediction,
                    "confidence": confidence,
                    "severity": severity,
                    "treatment": treatment,
                    "videos": videos,
                    "all_probs": all_probs,
                    "created_at": datetime.utcnow(),
                }
                insert_result = await scans_collection.insert_one(scan_doc)
                scan_id = str(insert_result.inserted_id)
                logger.info(f"[Predict] Saved scan {scan_id}")
        except Exception as db_error:
            logger.warning(f"[Predict] DB save failed: {db_error}")

        if scan_id:
            save_status = "saved"
        elif db_attempted:
            save_status = "offline"
        else:
            save_status = "guest"

        return {
            "id": scan_id or save_status,
            "prediction": prediction,
            "confidence": confidence,
            "severity": severity,
            "color": color,
            "treatment": treatment,
            "videos": videos,
            "saved": scan_id is not None,
            "save_status": save_status,
            "all_probs": all_probs,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Backend prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-scan")
async def save_scan(
    file: UploadFile = File(...),
    prediction: str = Form(...),
    confidence: float = Form(...),
    severity: str = Form(...),
    lang: str = Form("en"),
    authorization: str | None = Header(default=None),
):
    """
    Persist a scan that was already inferred on the frontend.
    Also fetches relevant YouTube videos and returns them.
    """
    user_id = None
    if authorization:
        try:
            payload = decode_access_token(authorization.replace("Bearer ", ""))
            user_id = payload.get("sub")
        except Exception:
            pass

    if lang not in {"en", "tw", "dag"}:
        lang = "en"

    try:
        filepath = save_image(file)
        treatment = get_treatment(prediction, lang)
        videos = await safe_get_videos(prediction, lang)

        result = await scans_collection.insert_one(
            {
                "user_id": user_id,
                "image_url": str(filepath),
                "prediction": prediction,
                "confidence": confidence,
                "severity": severity,
                "treatment": treatment,
                "videos": videos,
                "lang": lang,
                "created_at": datetime.utcnow(),
            }
        )

        logger.info(
            f"[SaveScan] Saved scan {result.inserted_id} with {len(videos)} videos"
        )
        return {
            "scan_id": str(result.inserted_id),
            "saved": True,
            "videos": videos,
        }
    except Exception as e:
        logger.exception("Save scan failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos")
async def get_videos_endpoint(prediction: str, lang: str = "en"):
    videos = await safe_get_videos(prediction, lang)
    logger.info(
        f"[VideosEndpoint] Returning {len(videos)} videos for {prediction} ({lang})"
    )
    return {"videos": videos}