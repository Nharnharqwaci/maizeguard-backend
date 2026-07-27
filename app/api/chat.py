import logging
import os
import tempfile
import base64
import subprocess
import asyncio
import re
from typing import Optional

import aiohttp
from fastapi import APIRouter, Header, Query, UploadFile, File
from pydantic import BaseModel
from groq import Groq
from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId

from app.core.database import db
from app.core.security import decode_access_token
from app.services.mms_tts_service import mms_tts
from app.services.mms_stt_service import mms_stt

router = APIRouter()
logger = logging.getLogger(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat_sessions = db["chat_sessions"]
chat_messages = db["chat_messages"]

# ── TRANSLATION SETUP ──
GHANA_NLP_API_KEY = os.getenv("GHANA_NLP_API_KEY")
KHAYA_BASE = "https://translation-api.ghananlp.org"

_nlp = None
if GHANA_NLP_API_KEY:
    try:
        from ghana_nlp import GhanaNLP
        _nlp = GhanaNLP(api_key=GHANA_NLP_API_KEY)
        logger.info("[Translation] GhanaNLP pip client initialized")
    except Exception as e:
        logger.warning(f"[Translation] Failed to init GhanaNLP pip client: {e}")
        _nlp = None


# ── AGRICULTURAL TERM NORMALIZATION (fixes GhanaNLP domain gaps) ──
AGRIC_TERM_MAP: dict[str, dict[str, str]] = {
   "tw": {
        "aburo": "maize",
        "aburoo": "maize",
        "aburoɔ": "maize",
        "atoko": "millet",
        "borɔdeɛ": "banana",
        "anuanua": "pests",
        "nkoae": "weeds",
        "ɔgyefuo": "fertilizer",
        "ntutu": "fertilizer",
        "asase": "soil",
        "nsuo": "water",
        "yare": "disease",
        "yareɛ": "disease",
        "awu": "dead",
        "kookoo": "cocoa",
        "nkosua": "eggs",
        "nam": "meat",
    },
    "dag": {
        "kpaligu": "maize",
        "kubiu": "water",
        "nyɔŋ": "disease",
        "kpakpuri": "fertilizer",
        "tihi": "pests",
        "nyɔŋa": "disease",
    },
}

_EN_TO_LOCAL: dict[str, dict[str, str]] = {
    lang: {v: k for k, v in pairs.items()}
    for lang, pairs in AGRIC_TERM_MAP.items()
}


def _normalize_terms(text: str, lang: str) -> str:
    """Replace local ag terms with English before translation."""
    if lang not in AGRIC_TERM_MAP or not text:
        return text
    for local_term, english_term in AGRIC_TERM_MAP[lang].items():
        pattern = re.compile(re.escape(local_term), re.IGNORECASE)
        text = pattern.sub(english_term, text)
    return text


def _localize_terms(text: str, lang: str) -> str:
    """Replace English ag terms with local terms after translation."""
    if lang not in _EN_TO_LOCAL or not text:
        return text
    for english_term, local_term in _EN_TO_LOCAL[lang].items():
        pattern = re.compile(r"\b" + re.escape(english_term) + r"\b", re.IGNORECASE)
        text = pattern.sub(local_term, text)
    return text


# ── Async translation helpers ──
async def _translate_pip(text: str, pair: str) -> Optional[str]:
    if not _nlp or not text:
        return None
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _nlp.translate(text, language_pair=pair))
        if result and result != text:
            return result
        return None
    except Exception as e:
        logger.warning(f"[Translation] pip failed for {pair}: {e}")
        return None


async def _translate_rest(text: str, pair: str) -> Optional[str]:
    if not GHANA_NLP_API_KEY or not text:
        return None
    try:
        source, target = pair.split("-")
    except ValueError:
        return None

    url = f"{KHAYA_BASE}/v1/translate"
    payload = {"text": text, "source": source, "target": target}
    headers = {
        "Authorization": f"Bearer {GHANA_NLP_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200:
                    body = await r.text()
                    logger.warning(f"[Translation] REST returned {r.status} for {pair}: {body[:200]}")
                    return None
                data = await r.json()
                result = data.get("translated_text") or data.get("translation") or data.get("result")
                if result and result != text:
                    return result
                return None
    except asyncio.TimeoutError:
        logger.warning(f"[Translation] REST timeout for {pair}")
        return None
    except Exception as e:
        logger.warning(f"[Translation] REST failed for {pair}: {e}")
        return None


async def translate(text: str, pair: str) -> Optional[str]:
    result = await _translate_pip(text, pair)
    if result and result != text:
        return result
    result = await _translate_rest(text, pair)
    if result and result != text:
        return result
    return None


async def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang == "en" or not text or not text.strip():
        return text
    normalized = _normalize_terms(text, source_lang)
    result = await translate(normalized, f"{source_lang}-en")
    return result if result else normalized


async def translate_from_english(text: str, target_lang: str) -> str:
    if target_lang == "en" or not text or not text.strip():
        return text
    result = await translate(text, f"en-{target_lang}")
    if result:
        return _localize_terms(result, target_lang)
    return text


# ── Audio conversion: webm -> wav ──
def convert_webm_to_wav(webm_path: str) -> Optional[str]:
    wav_path = webm_path.replace(".webm", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-i", webm_path, "-ar", "16000", "-ac", "1", "-y", wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return wav_path
    except Exception as e:
        logger.error(f"[Audio] ffmpeg failed: {e}")
        return None


# ── STT: Meta MMS (Twi + English) → Khaya AI (Dagbani fallback) ──
async def _khaya_stt(audio_path: str, lang: str) -> Optional[str]:
    if not GHANA_NLP_API_KEY:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            with open(audio_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("audio", f, filename=os.path.basename(audio_path), content_type="audio/wav")
                data.add_field("language", lang)
                async with session.post(
                    f"{KHAYA_BASE}/v1/asr",
                    data=data,
                    headers={"Authorization": f"Bearer {GHANA_NLP_API_KEY}"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    if r.status != 200:
                        body = await r.text()
                        logger.warning(f"[STT] Khaya returned {r.status}: {body[:200]}")
                        return None
                    result_data = await r.json()
                    return result_data.get("text") or result_data.get("transcript") or result_data.get("result")
    except Exception as e:
        logger.warning(f"[STT] Khaya fallback failed: {e}")
        return None


async def _ghana_nlp_stt(audio_path: str, lang: str) -> Optional[str]:
    if not _nlp:
        return None
    try:
        loop = asyncio.get_event_loop()
        if hasattr(_nlp, "speech_to_text"):
            return await loop.run_in_executor(None, lambda: _nlp.speech_to_text(audio_path, language=lang))
        elif hasattr(_nlp, "stt"):
            return await loop.run_in_executor(None, lambda: _nlp.stt(audio_path, language=lang))
        elif hasattr(_nlp, "transcribe"):
            return await loop.run_in_executor(None, lambda: _nlp.transcribe(audio_path, language=lang))
        return None
    except Exception as e:
        logger.warning(f"[STT] GhanaNLP pip STT failed: {e}")
        return None


async def speech_to_text(audio_bytes: bytes, lang: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        webm_path = tmp.name

    try:
        wav_path = convert_webm_to_wav(webm_path)
        if not wav_path or not os.path.exists(wav_path):
            wav_path = webm_path

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, mms_stt.transcribe, wav_path, lang)
        if text:
            logger.info(f"[STT] MMS succeeded: {text[:80]}")
            return text
        logger.info(f"[STT] MMS returned empty for {lang}, trying Khaya fallback...")

        text = await _khaya_stt(wav_path, lang)
        if text:
            return text

        text = await _ghana_nlp_stt(wav_path, lang)
        if text:
            return text

        return ""
    finally:
        for p in [webm_path, webm_path.replace(".webm", ".wav")]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


# ── TTS: Meta MMS (Twi, English) → Khaya AI (Dagbani fallback) ──
async def _khaya_tts(text: str, lang: str) -> Optional[bytes]:
    if not GHANA_NLP_API_KEY:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{KHAYA_BASE}/v1/tts",
                headers={
                    "Authorization": f"Bearer {GHANA_NLP_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"text": text, "language": lang},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                if r.status != 200:
                    body = await r.text()
                    logger.warning(f"[TTS] Khaya returned {r.status}: {body[:200]}")
                    return None
                content_type = r.headers.get("Content-Type", "")
                if "json" in content_type:
                    data = await r.json()
                    audio_b64 = data.get("audio") or data.get("audio_base64") or data.get("result")
                    if audio_b64:
                        return base64.b64decode(audio_b64)
                    return None
                return await r.read()
    except Exception as e:
        logger.warning(f"[TTS] Khaya fallback failed: {e}")
        return None


async def text_to_speech(text: str, lang: str) -> Optional[bytes]:
    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, mms_tts.synthesize, text, lang)
    if audio:
        return audio

    if lang in ("dag",):
        logger.info(f"[TTS] MMS doesn't support {lang}, trying Khaya AI TTS...")
        audio = await _khaya_tts(text, lang)
        if audio:
            return audio

    return None


DISEASE_CONTEXT: dict[str, str] = {
    "Common_Rust": """The farmer's maize leaf has been diagnosed with Common Rust (Puccinia sorghi).
Common Rust is a fungal disease causing orange-red pustules on both leaf surfaces.
It spreads rapidly in cool humid conditions (16-25°C) and can cause yield losses of 12-75%.
Recommended treatments: azoxystrobin, propiconazole, or mancozeb fungicides.
Resistant varieties and crop rotation are the best long-term strategies.""",

    "Gray_Leaf_Spot": """The farmer's maize leaf has been diagnosed with Gray Leaf Spot (Cercospora zeae-maydis).
Gray Leaf Spot causes rectangular gray-tan lesions between leaf veins.
It thrives in warm humid conditions and can cause 20-50% yield losses if untreated.
Strobilurin or triazole fungicides applied early are effective.
Crop rotation with soybeans or cowpea and improved drainage are key management strategies.""",

    "Healthy": """The farmer's maize leaf has been diagnosed as Healthy.
The crop shows no signs of disease.
The farmer should maintain good agronomic practices including proper NPK fertilization (60-40-40 kg/ha),
adequate plant spacing (75cm between rows, 25cm within rows), and weekly field monitoring.
Preventive measures against common maize diseases in Ghana should be discussed.""",

    "MSV": """The farmer's maize leaf has been diagnosed with Maize Streak Virus (MSV).
MSV is a viral disease endemic in sub-Saharan Africa spread by leafhoppers (Cicadulina spp.).
It causes yellow streaks or mosaic patterns on leaves and has no cure once a plant is infected.
Immediate action: remove infected plants, control leafhoppers with imidacloprid or thiamethoxam,
plant MSV-resistant varieties like SAMMAZ 14 or SAMMAZ 15 next season.
This must be reported to agricultural extension officers.""",

    "Northern_Leaf_Blight": """The farmer's maize leaf has been diagnosed with Northern Leaf Blight (Exserohilum turcicum).
Northern Leaf Blight causes long cigar-shaped tan lesions on maize leaves.
It is favoured by moderate temperatures and prolonged leaf wetness and can cause 30-50% yield losses.
Propiconazole, azoxystrobin, or pyraclostrobin fungicides applied early are effective.
Crop residue management and resistant hybrids are important for long-term control.""",

    "Southern_Leaf_Blight": """The farmer's maize leaf has been diagnosed with Southern Leaf Blight (Cochliobolus heterostrophus).
Southern Leaf Blight causes small tan lesions with brown borders — most severe in warm humid regions like coastal Ghana.
Immediate application of strobilurin or triazole fungicides is required.
Infected material must be burned — never composted.
Potassium nutrition is important for building plant resistance.""",

    "Uncertain": """The system could not confidently classify the farmer's maize leaf image.
The image may not be a maize leaf, may be blurry, or the lighting may be poor.
Guide the farmer on how to take a better photo for accurate diagnosis.
If they describe disease symptoms verbally, help identify the likely disease based on their description.""",
}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    treatment: Optional[list[str]] = None
    session_id: Optional[str] = None
    new_session: bool = False
    language: str = "en"


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


def get_optional_user_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ").strip()
    payload = decode_access_token(token)
    if not payload:
        return None
    return payload.get("sub")


def build_system_prompt(
    prediction: Optional[str],
    confidence: Optional[float],
    treatment: Optional[list[str]],
    language: str,
) -> str:
    disease_context = ""
    if prediction:
        disease_context = DISEASE_CONTEXT.get(prediction, DISEASE_CONTEXT["Uncertain"])

    treatment_context = ""
    if treatment:
        lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(treatment))
        treatment_context = f"\nRecommended treatments already given to the farmer:\n{lines}"

    confidence_str = ""
    if prediction and confidence:
        confidence_str = f"\nConfidence level: {confidence:.1f}%"

    lang_instruction = ""
    if language == "tw":
        lang_instruction = (
            "The farmer is communicating in Twi (Akan). "
            "You must respond in English — the backend will translate your reply to Twi automatically. "
            "Keep your English clear, simple, and culturally appropriate for Ghanaian smallholder farmers."
        )
    elif language == "dag":
        lang_instruction = (
            "The farmer is communicating in Dagbani. "
            "You must respond in English — the backend will translate your reply to Dagbani automatically. "
            "Keep your English clear, simple, and culturally appropriate for Ghanaian smallholder farmers."
        )
    else:
        lang_instruction = "The farmer is communicating in English. Respond in English."

    return f"""You are an expert Agricultural Extension Officer specializing in maize or aburoo farming in Ghana and West Africa.
You have deep knowledge of maize diseases, pest management, soil health, fertilization, and sustainable farming practices.

{f"CURRENT DIAGNOSIS CONTEXT:{chr(10)}{disease_context}" if disease_context else ""}
{confidence_str}
{treatment_context}

YOUR ROLE AND PERSONALITY:
- You are a helpful, knowledgeable, and empathetic agricultural officer
- You speak clearly and practically, avoiding overly technical jargon
- You give specific, actionable advice relevant to Ghana and West Africa
- You mention locally available products and resources when possible
- You always consider the economic constraints of smallholder farmers
- You encourage farmers and acknowledge their challenges
- You recommend contacting local MOFA extension officers for severe cases
- Keep responses concise but complete — 3 to 6 sentences unless more detail is needed
- If asked about something outside agriculture, politely redirect to farming topics

{lang_instruction}

IMPORTANT: Base your responses on the diagnosis context above. Give tailored advice based on the detected disease."""


def make_title(message: str, prediction: Optional[str]) -> str:
    message = (message or "").strip()
    if message:
        return message[:40] + ("…" if len(message) > 40 else "")
    if prediction:
        return prediction.replace("_", " ")
    return "New conversation"


@router.post("/chat")
async def chat(request: ChatRequest, authorization: Optional[str] = Header(None)):
    user_id = get_optional_user_id(authorization)
    lang = request.language if request.language in ("en", "tw", "dag") else "en"

    try:
        english_input = await translate_to_english(request.message, lang)
        logger.info(f"[Chat] lang={lang} translated_input_preview={english_input[:80]!r}")

        session = None
        session_id: Optional[str] = None

        if user_id:
            if request.session_id:
                try:
                    session = await chat_sessions.find_one({
                        "_id": ObjectId(request.session_id),
                        "user_id": user_id,
                    })
                except InvalidId:
                    session = None

            if not session and not request.new_session:
                latest_list = await chat_sessions.find(
                    {"user_id": user_id}
                ).sort("updated_at", -1).to_list(length=1)

                if latest_list and latest_list[0].get("prediction") == request.prediction:
                    session = latest_list[0]

            if not session:
                now = datetime.now(timezone.utc)
                new_doc = {
                    "user_id": user_id,
                    "title": make_title(request.message, request.prediction),
                    "prediction": request.prediction,
                    "language": lang,
                    "created_at": now,
                    "updated_at": now,
                }
                insert_result = await chat_sessions.insert_one(new_doc)
                session = {**new_doc, "_id": insert_result.inserted_id}

            session_id = str(session["_id"])

            past = await chat_messages.find(
                {"session_id": session_id}
            ).sort("timestamp", 1).to_list(length=40)

            history = [{"role": m["role"], "content": m["content_en"]} for m in past]
        else:
            history = []
            for m in request.history:
                en_content = await translate_to_english(m.content, lang) if lang != "en" else m.content
                history.append({"role": m.role, "content": en_content})

        history.append({"role": "user", "content": english_input})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": build_system_prompt(
                        request.prediction, request.confidence, request.treatment, lang
                    )
                },
                *history
            ],
            max_tokens=1000,
            temperature=0.7,
        )

        english_reply = response.choices[0].message.content
        display_reply = await translate_from_english(english_reply, lang)
        logger.info(f"[Chat] lang={lang} translated_reply_preview={display_reply[:80]!r}")

        if user_id and session_id:
            now = datetime.now(timezone.utc)
            await chat_messages.insert_many([
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "role": "user",
                    "content_en": english_input,
                    "content_display": request.message,
                    "timestamp": now,
                },
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "role": "assistant",
                    "content_en": english_reply,
                    "content_display": display_reply,
                    "timestamp": now,
                },
            ])
            await chat_sessions.update_one(
                {"_id": session["_id"]},
                {"$set": {"updated_at": now}},
            )

        return {"reply": display_reply, "session_id": session_id, "language": lang}

    except Exception as e:
        logger.exception(f"[CHAT ERROR] lang={lang} msg_preview={request.message[:60]!r} error={e}")
        fallback = (
            "I'm sorry, I'm having trouble responding right now. Please try again in a moment."
            if lang == "en"
            else "Kafra, seesei mintumi mmua wo. Mepa wo kyɛw, san bɔ mmɔden yɛ bio."
            if lang == "tw"
            else "N-yɛli, n ti n-gbaŋ. Chɛma ka dii yɛl' pahi."
        )
        return {"reply": fallback, "session_id": None, "language": lang}


@router.post("/chat/stt")
async def stt_endpoint(
    audio: UploadFile = File(...),
    language: str = Query("tw"),
):
    audio_bytes = await audio.read()
    text = await speech_to_text(audio_bytes, language)
    return {"text": text or "", "language": language}


@router.post("/chat/tts")
async def tts_endpoint(request: TTSRequest):
    audio = await text_to_speech(request.text, request.language)
    if audio is None:
        return {"audio": None, "error": "TTS unavailable"}
    return {"audio": base64.b64encode(audio).decode("utf-8"), "language": request.language}


@router.get("/chat/tts-health")
async def tts_health():
    return {
        "tts": mms_tts.health(),
        "stt": mms_stt.health(),
        "ghana_nlp_key_set": bool(GHANA_NLP_API_KEY),
    }


@router.get("/chat/sessions")
async def list_sessions(authorization: Optional[str] = Header(None)):
    user_id = get_optional_user_id(authorization)
    if not user_id:
        return {"sessions": []}

    sessions = await chat_sessions.find(
        {"user_id": user_id}
    ).sort("updated_at", -1).to_list(length=50)

    return {
        "sessions": [
            {
                "id": str(s["_id"]),
                "title": s.get("title", "Conversation"),
                "prediction": s.get("prediction"),
                "language": s.get("language", "en"),
                "updated_at": s["updated_at"].isoformat(),
            }
            for s in sessions
        ]
    }


@router.get("/chat/history")
async def get_chat_history(
    session_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    user_id = get_optional_user_id(authorization)
    if not user_id:
        return {"messages": [], "session_id": None, "prediction": None, "language": "en"}

    session = None
    if session_id:
        try:
            session = await chat_sessions.find_one({
                "_id": ObjectId(session_id),
                "user_id": user_id,
            })
        except InvalidId:
            session = None
    else:
        latest_list = await chat_sessions.find(
            {"user_id": user_id}
        ).sort("updated_at", -1).to_list(length=1)
        session = latest_list[0] if latest_list else None

    if not session:
        return {"messages": [], "session_id": None, "prediction": None, "language": "en"}

    resolved_id = str(session["_id"])
    past = await chat_messages.find(
        {"session_id": resolved_id}
    ).sort("timestamp", 1).to_list(length=100)

    return {
        "session_id": resolved_id,
        "prediction": session.get("prediction"),
        "language": session.get("language", "en"),
        "messages": [
            {
                "role": m["role"],
                "content": m.get("content_display", m.get("content_en", "")),
                "content_en": m.get("content_en", ""),
                "timestamp": m["timestamp"].isoformat(),
            }
            for m in past
        ],
    }


@router.delete("/chat/history")
async def delete_chat_history(
    session_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    user_id = get_optional_user_id(authorization)
    if not user_id:
        return {"status": "no_user"}

    if session_id:
        try:
            oid = ObjectId(session_id)
        except InvalidId:
            return {"status": "invalid_session"}
        await chat_messages.delete_many({"session_id": session_id, "user_id": user_id})
        await chat_sessions.delete_one({"_id": oid, "user_id": user_id})
        return {"status": "cleared", "session_id": session_id}

    session_ids = [
        str(s["_id"])
        for s in await chat_sessions.find({"user_id": user_id}).to_list(length=1000)
    ]
    if session_ids:
        await chat_messages.delete_many({"session_id": {"$in": session_ids}, "user_id": user_id})
    await chat_sessions.delete_many({"user_id": user_id})
    return {"status": "cleared_all"}
