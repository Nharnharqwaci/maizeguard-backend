from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Header
from datetime import datetime
import logging
import math
import asyncio

from app.services.image_service import save_image
from app.services.yolo_service import run_inference
from app.services.youtube_service import get_videos
from app.core.database import scans_collection
from app.core.security import decode_access_token

logger = logging.getLogger(__name__)
router = APIRouter()


CONFIDENCE_THRESHOLD = 70.0
ENTROPY_THRESHOLD = 0.5


def compute_normalized_entropy(probs: dict[str, float]) -> float:
    """probs values are fractions (0-1), NOT percentages.
    Returns a value from 0 (one class fully dominant) to 1 (uniform across
    all classes — maximum uncertainty)."""
    values = [p for p in probs.values() if p > 0]
    if len(values) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in values)
    max_entropy = math.log(len(probs))
    return entropy / max_entropy if max_entropy > 0 else 0.0


# Multilingual treatment dictionary
# Language codes: en = English, tw = Twi, dag = Dagbani
TREATMENTS: dict[str, dict[str, list[str]]] = {
    "Common_Rust": {
        "en": [
            "Common Rust (Puccinia sorghi) detected — fungal disease requiring prompt action.",
            "Apply fungicides containing azoxystrobin, propiconazole, or mancozeb.",
            "Remove heavily infected leaves.",
            "Plant rust-resistant maize varieties.",
            "Rotate maize with legumes.",
            "Maintain proper spacing for airflow.",
            "Monitor fields every week."
        ],
        "tw": [
            "Common Rust (Puccinia sorghi) ahyia — ɛyɛ fungus yaree a ɛhia sɛ wobɛyɛ no ntɛm.",
            "Fa fungicides a ɛwɔ azoxystrobin, propiconazole, anaa mancozeb gu so.",
            "Yi ahaban a yaree no atra mu no nyinaa.",
            "Dua aburow a ɛnnyɛ rust yaree.",
            "Dua aburow ne nkesua nnɔbae.",
            "Siesie anammɔn a ɛfata ma mframa.",
            "Hwɛ afuo no abɔso biara."
        ],
        "dag": [
            "Common Rust (Puccinia sorghi) n-nya — fungus yɛl' ni sɔŋsim pampam.",
            "Zaŋ fungicides ni azoxystrobin, propiconazole, bee mancozeb gu so.",
            "Yihi kpamli ni yɛl' atra mu.",
            "Dug maize ni biɛla rust yɛl'.",
            "Dug maize ni nkesua sal'.",
            "Kpaŋsim anammɔn ni mframa tuuli.",
            "Kpaŋsim afuo nyɔŋɔ."
        ]
    },
    "Gray_Leaf_Spot": {
        "en": [
            "Gray Leaf Spot detected.",
            "Apply recommended foliar fungicides.",
            "Improve drainage.",
            "Practice crop rotation.",
            "Destroy infected crop residue.",
            "Use resistant maize varieties."
        ],
        "tw": [
            "Grey Leaf Spot ahyia.",
            "Fa foliar fungicides a wɔakyerɛw gu so.",
            "Siesie nsu a ɛbɛfiri fam no.",
            "Dua nnɔbae ahorow.",
            "Sɛe nnɔbae a yaree no atra mu no.",
            "Fa aburow a ɛnnyɛ yaree no."
        ],
        "dag": [
            "Gray Leaf Spot n-nya.",
            "Zaŋ foliar fungicides ni wɔkyɛn gu so.",
            "Kpaŋsim nsu niŋsim.",
            "Dug sal' a zaa.",
            "Sɛɛ sal' ni yɛl' atra mu.",
            "Zaŋ maize ni biɛla yɛl'."
        ]
    },
    "Healthy": {
        "en": [
            "Excellent! No disease detected.",
            "Continue regular crop monitoring.",
            "Maintain proper fertilization.",
            "Keep weeds under control.",
            "Inspect plants weekly.",
            "Keep following good agronomic practices."
        ],
        "tw": [
            "Ayɛ papa! Yaree biara nni hɔ.",
            "Toa so hwɛ nnɔbae no.",
            "Siesie aduane a ɛfata.",
            "Hwɛ sɛ nwura no nni hɔ.",
            "Hwɛ nnɔbae no abɔso biara.",
            "Toa so di kuayɛ a ɛyɛ yie no so."
        ],
        "dag": [
            "Chɛli! Yɛl' biɛla biɛla.",
            "Tuuli kpamli sal' kpaŋsim.",
            "Kpaŋsim aduane niŋsim.",
            "Kpaŋsim nwura.",
            "Kpaŋsim sal' nyɔŋɔ.",
            "Tuuli kuayɛ ni nyɛ yɛn."
        ]
    },
    "MSV": {
        "en": [
            "Maize Streak Virus detected.",
            "Remove severely infected plants immediately.",
            "Control leafhopper vectors.",
            "Plant resistant maize varieties.",
            "Avoid continuous maize cropping.",
            "There is no cure for infected plants.",
            "Prevent spread to healthy plants."
        ],
        "tw": [
            "Maize Streak Virus ahyia.",
            "Yi nnɔbae a yaree no atra mu no ntɛm.",
            "Kɔ leafhopper a wɔde yaree no kɔma nnɔbae no so.",
            "Dua aburow a ɛnnyɛ yaree.",
            "Nyɛ aburow daa.",
            "Yaree no nni ayaresa.",
            "Si kwan ma yaree no ankɔ nnɔbae a apɔwmuden wom so."
        ],
        "dag": [
            "Maize Streak Virus n-nya.",
            "Yihi sal' ni yɛl' atra mu pampam.",
            "Kpaŋsim leafhopper ni yɛl' to sal' so.",
            "Dug maize ni biɛla yɛl'.",
            "Dolima maize daa.",
            "Yɛl' ayaresa biɛla.",
            "Sɔŋsim yɛl' ni n-kɔ sal' ni kpalim zaa so."
        ]
    },
    "Northern_Leaf_Blight": {
        "en": [
            "Northern Leaf Blight detected.",
            "Apply fungicides early.",
            "Improve field airflow.",
            "Destroy infected residues.",
            "Rotate crops.",
            "Plant resistant hybrids."
        ],
        "tw": [
            "Northern Leaf Blight ahyia.",
            "Fa fungicides gu so ntɛm.",
            "Ma mframa nya kwan wɔ afuo no mu.",
            "Sɛe nnɔbae a yaree no atra mu no.",
            "Dua nnɔbae ahorow.",
            "Dua nnɔbae a ɛnnyɛ yaree."
        ],
        "dag": [
            "Northern Leaf Blight n-nya.",
            "Zaŋ fungicides gu so pampam.",
            "Kpaŋsim mframa wɔ afuo ni.",
            "Sɛɛ sal' ni yɛl' atra mu.",
            "Dug sal' a zaa.",
            "Dug sal' ni biɛla yɛl'."
        ]
    },
    "Southern_Leaf_Blight": {
        "en": [
            "Southern Leaf Blight detected.",
            "Apply fungicides immediately.",
            "Remove infected plants.",
            "Improve drainage.",
            "Use certified seed.",
            "Rotate crops."
        ],
        "tw": [
            "Southern Leaf Blight ahyia.",
            "Fa fungicides gu so ntɛm.",
            "Yi nnɔbae a yaree no atra mu no.",
            "Siesie nsu a ɛbɛfiri fam no.",
            "Fa nnuaba a wɔahwɛ so.",
            "Dua nnɔbae ahorow."
        ],
        "dag": [
            "Southern Leaf Blight n-nya.",
            "Zaŋ fungicides gu so pampam.",
            "Yihi sal' ni yɛl' atra mu.",
            "Kpaŋsim nsu niŋsim.",
            "Zaŋ nnuaba ni wɔkpaŋsi.",
            "Dug sal' a zaa."
        ]
    },
    "Uncertain": {
        "en": [
            "The image could not be confidently classified.",
            "Please upload a clearer maize leaf image.",
            "Ensure the leaf occupies most of the picture.",
            "Take the picture in natural daylight.",
            "Avoid blurry images.",
            "Make sure the image actually shows a maize leaf."
        ],
        "tw": [
            "Yɛnntumi ankyekyɛ mfonini no.",
            "Yɛsrɛ wo twe aburow ahaban mfonini a ɛyɛ kyerɛ.",
            "Hwɛ sɛ ahaban no wɔ mfonini no mu.",
            "Twe mfonini no wɔ awia mu.",
            "Nyɛ mfonini a ɛnnyɛ kyerɛ.",
            "Hwɛ sɛ mfonini no kyerɛ aburow ahaban."
        ],
        "dag": [
            "N-tum kpari nimli maa.",
            "Yɛn zaŋ maize kpamli nimli ni nyɛ yɛn.",
            "Kpaŋsim kpamli n-nya nimli ni pahi.",
            "Twa nimli wɔ awia ni.",
            "Dolima nimli ni biɛla.",
            "Kpaŋsim nimli n-nya maize kpamli."
        ]
    }
}


def get_treatment(prediction: str, lang: str = "en") -> list[str]:
    """Get treatment recommendations in the specified language.
    Falls back to English if the language is not available."""
    disease_treatments = TREATMENTS.get(prediction, TREATMENTS["Uncertain"])
    if lang in disease_treatments:
        return disease_treatments[lang]
    return disease_treatments.get("en", disease_treatments.get("tw", disease_treatments.get("dag", [])))


# Multilingual severity labels
SEVERITY_LABELS: dict[str, dict[str, str]] = {
    "Healthy": {
        "en": "none",
        "tw": "biara nni hɔ",
        "dag": "biɛla"
    },
    "Common_Rust": {
        "en": "medium",
        "tw": "mfimfini",
        "dag": "mfimfini"
    },
    "Gray_Leaf_Spot": {
        "en": "medium",
        "tw": "mfimfini",
        "dag": "mfimfini"
    },
    "MSV": {
        "en": "high",
        "tw": "koraa",
        "dag": "koraa"
    },
    "Northern_Leaf_Blight": {
        "en": "high",
        "tw": "koraa",
        "dag": "koraa"
    },
    "Southern_Leaf_Blight": {
        "en": "critical",
        "tw": "den",
        "dag": "den"
    },
    "Uncertain": {
        "en": "low",
        "tw": "tia",
        "dag": "tia"
    }
}


def get_severity(prediction: str, lang: str = "en") -> str:
    """Get severity label. Returns English code for frontend styling,
    but could be extended to return translated labels."""
    return {
        "Healthy": "none",
        "Common_Rust": "medium",
        "Gray_Leaf_Spot": "medium",
        "MSV": "high",
        "Northern_Leaf_Blight": "high",
        "Southern_Leaf_Blight": "critical",
        "Uncertain": "low"
    }.get(prediction, "low")


def get_color(prediction: str) -> str:
    return {
        "Healthy": "green",
        "Common_Rust": "orange",
        "Gray_Leaf_Spot": "purple",
        "MSV": "red",
        "Northern_Leaf_Blight": "amber",
        "Southern_Leaf_Blight": "rose",
        "Uncertain": "yellow"
    }.get(prediction, "yellow")


async def safe_get_videos(prediction: str, timeout: float = 6.0):
    """Fetch YouTube videos with a strict timeout to avoid blocking predictions."""
    try:
        # Run get_videos in a thread with timeout
        loop = asyncio.get_event_loop()
        videos = await asyncio.wait_for(
            loop.run_in_executor(None, get_videos, prediction),
            timeout=timeout
        )
        return videos
    except asyncio.TimeoutError:
        logger.warning(f"YouTube videos timed out for {prediction} after {timeout}s")
        return []
    except Exception as e:
        logger.warning(f"YouTube videos failed for {prediction}: {e}")
        return []


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    user_id: str = Form(None),
    lang: str = Form("en")  # Language parameter: en, tw, or dag
):
    resolved_user_id = None

    if authorization:
        try:
            token = authorization.replace("Bearer ", "")
            payload = decode_access_token(token)
            resolved_user_id = payload.get("sub")
            logger.info(f"Token decoded - user_id: {resolved_user_id}")
        except Exception as e:
            logger.warning(f"Failed to decode token: {e}")
            resolved_user_id = None

    final_user_id = resolved_user_id or user_id

    # Validate language parameter
    valid_langs = {"en", "tw", "dag"}
    if lang not in valid_langs:
        lang = "en"
        logger.info(f"Invalid language requested, defaulting to 'en'")

    try:
        if file.content_type not in [
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp"
        ]:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, PNG and WEBP images are supported."
            )

        filepath = save_image(file)

        result = run_inference(filepath)

        prediction = result["class"]

        confidence = round(
            result["confidence"] * 100,
            2
        )

        all_probs = result.get(
            "all_probs",
            {}
        )

        valid_classes = [
            "Common_Rust",
            "Gray_Leaf_Spot",
            "Healthy",
            "MSV",
            "Northern_Leaf_Blight",
            "Southern_Leaf_Blight"
        ]

        if prediction not in valid_classes:
            prediction = "Uncertain"
            logger.info(f"Rejected — model returned unrecognized class: {result.get('class')}")
        else:
            normalized_entropy = compute_normalized_entropy(all_probs)

            if confidence < CONFIDENCE_THRESHOLD:
                logger.info(
                    f"Rejected — confidence {confidence}% below threshold "
                    f"{CONFIDENCE_THRESHOLD}% (was: {prediction})"
                )
                prediction = "Uncertain"
            elif normalized_entropy > ENTROPY_THRESHOLD:
                logger.info(
                    f"Rejected — entropy {normalized_entropy:.2f} above threshold "
                    f"{ENTROPY_THRESHOLD} (was: {prediction}, confidence: {confidence}%)"
                )
                prediction = "Uncertain"

        treatment = get_treatment(prediction, lang)

        severity = get_severity(prediction)

        color = get_color(prediction)

        # Fetch videos with timeout protection
        videos = await safe_get_videos(prediction)

        scan_id = None
        db_attempted = False

        try:
            if final_user_id:
                scan_doc = {
                    "user_id": final_user_id,
                    "image_url": str(filepath),
                    "prediction": prediction,
                    "confidence": confidence,
                    "severity": severity,
                    "treatment": treatment,
                    "videos": videos,
                    "lang": lang,  # <-- FIX: save the language used for this scan
                    "all_probs": {
                        k: round(v * 100, 2)
                        for k, v in all_probs.items()
                    },
                    "created_at": datetime.utcnow()
                }

                insert_result = await scans_collection.insert_one(scan_doc)

                scan_id = str(insert_result.inserted_id)

                logger.info(
                    f"Saved Scan {scan_id} (lang={lang})"
                )

        except Exception as db_error:

            logger.warning(
                f"Database save failed: {db_error}"
            )
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
            "all_probs": {
                k: round(v * 100, 2)
                for k, v in all_probs.items()
            },
            "lang": lang  # Return the language used for the response
        }

    except HTTPException:
        raise

    except Exception as e:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
