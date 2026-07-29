# app/services/mms_tts_service.py
import io
import logging
import wave
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import VitsModel, AutoTokenizer
    _MMS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[MMS-TTS] Missing dependencies (torch, transformers): {e}")
    _MMS_AVAILABLE = False

# Only languages with confirmed HuggingFace models
MMS_LANG_MAP = {
    "en": "eng",
    "tw": "aka",
}

MMS_UNSUPPORTED = {"dag"}


class MMSTTSService:
    """Local Meta MMS Text-to-Speech for Twi and English."""

    def __init__(self):
        self._models: dict[str, tuple] = {}
        self._device = "cpu"
        if _MMS_AVAILABLE and torch is not None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[MMS-TTS] Init. Available={_MMS_AVAILABLE}, Device={self._device}")

    def preload(self, lang_codes: Optional[list[str]] = None):
        """Pre-load models into memory. Call this at startup if you have enough RAM."""
        if not _MMS_AVAILABLE:
            logger.error("[MMS-TTS] Cannot preload — torch/transformers not installed")
            return
        for code in (lang_codes or list(MMS_LANG_MAP.keys())):
            if code in MMS_UNSUPPORTED:
                logger.info(f"[MMS-TTS] Skipping {code} — no MMS model exists. Will use Khaya fallback.")
                continue
            self._load_model(code)

    def _load_model(self, lang_code: str) -> Optional[tuple]:
        if not _MMS_AVAILABLE:
            return None
        if lang_code in self._models:
            return self._models[lang_code]
        if lang_code in MMS_UNSUPPORTED:
            return None

        mms_code = MMS_LANG_MAP.get(lang_code)
        if not mms_code:
            logger.warning(f"[MMS-TTS] Unsupported app language: {lang_code}")
            return None

        model_id = f"facebook/mms-tts-{mms_code}"
        try:
            logger.info(f"[MMS-TTS] Loading {model_id}...")
            model = VitsModel.from_pretrained(model_id).to(self._device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model.eval()
            self._models[lang_code] = (model, tokenizer)
            logger.info(f"[MMS-TTS] {model_id} loaded OK on {self._device}")
            return self._models[lang_code]
        except Exception as e:
            logger.error(f"[MMS-TTS] FAILED to load {model_id}: {e}")
            return None

    def synthesize(self, text: str, lang_code: str) -> Optional[bytes]:
        if not _MMS_AVAILABLE:
            logger.warning("[MMS-TTS] synthesize called but dependencies missing")
            return None
        if not text or not text.strip():
            logger.warning("[MMS-TTS] Empty text received")
            return None
        if lang_code in MMS_UNSUPPORTED:
            logger.info(f"[MMS-TTS] {lang_code} not supported by MMS — returning None for Khaya fallback")
            return None

        loaded = self._load_model(lang_code)
        if not loaded:
            logger.warning(f"[MMS-TTS] No model loaded for lang={lang_code}")
            return None

        model, tokenizer = self._models[lang_code]

        try:
            inputs = tokenizer(text.strip(), return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model(**inputs).waveform

            waveform = output.cpu().numpy()
            if waveform.ndim > 1:
                waveform = waveform.squeeze()

            samples = (waveform * 32767).astype("int16")

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(model.config.sampling_rate)
                wav_file.writeframes(samples.tobytes())

            buffer.seek(0)
            wav_bytes = buffer.read()
            logger.info(f"[MMS-TTS] Synthesized {len(wav_bytes)} bytes for {lang_code}")
            return wav_bytes

        except Exception as e:
            logger.error(f"[MMS-TTS] Synthesis failed for {lang_code}: {e}")
            return None

    def health(self) -> dict:
        return {
            "available": _MMS_AVAILABLE,
            "device": self._device,
            "loaded_models": list(self._models.keys()),
            "supported_languages": list(MMS_LANG_MAP.keys()),
            "unsupported_languages": list(MMS_UNSUPPORTED),
        }


# Singleton instance — imported lazily by chat.py
mms_tts = MMSTTSService()