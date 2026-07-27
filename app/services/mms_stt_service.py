# app/services/mms_stt_service.py
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import numpy as np
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    import librosa
    _MMS_STT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[MMS-STT] Missing dependencies (torch, transformers, librosa): {e}")
    _MMS_STT_AVAILABLE = False

# MMS STT supports 'aka' (Akan/Twi) and 'eng' but NOT 'dag' (Dagbani)
MMS_STT_LANG_MAP = {
    "en": "eng",
    "tw": "aka",
}

MMS_STT_UNSUPPORTED = {"dag"}


class MMSSTTService:
    """Local Meta MMS Speech-to-Text. Supports Twi (aka) and English (eng).
    Dagbani STT falls back to Khaya AI — MMS does not have a dag adapter."""

    def __init__(self):
        self._models: dict[str, tuple] = {}
        self._base_model_loaded = False
        self._device = "cpu"
        self._base_processor = None
        self._base_model = None
        if _MMS_STT_AVAILABLE and torch is not None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[MMS-STT] Init. Available={_MMS_STT_AVAILABLE}, Device={self._device}")

    def preload(self, lang_codes: Optional[list[str]] = None):
        if not _MMS_STT_AVAILABLE:
            logger.error("[MMS-STT] Cannot preload — dependencies missing")
            return
        for code in (lang_codes or ["en", "tw"]):
            if code in MMS_STT_UNSUPPORTED:
                logger.info(f"[MMS-STT] Skipping preload for {code} — not supported. Will use Khaya fallback.")
                continue
            self._load_model(code)

    def _load_base(self) -> bool:
        if not _MMS_STT_AVAILABLE:
            return False
        if self._base_model_loaded:
            return True
        try:
            model_id = "facebook/mms-1b-all"
            logger.info(f"[MMS-STT] Loading base model {model_id} (~2.5 GB)...")
            self._base_processor = AutoProcessor.from_pretrained(model_id)
            self._base_model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self._device)
            self._base_model.eval()
            self._base_model_loaded = True
            logger.info(f"[MMS-STT] Base model loaded on {self._device}")
            return True
        except Exception as e:
            logger.error(f"[MMS-STT] FAILED to load base model: {e}")
            return False

    def _load_model(self, lang_code: str) -> Optional[tuple]:
        if not _MMS_STT_AVAILABLE:
            return None
        if lang_code in self._models:
            return self._models[lang_code]
        if lang_code in MMS_STT_UNSUPPORTED:
            logger.info(f"[MMS-STT] {lang_code} not supported — skipping adapter load")
            return None

        mms_code = MMS_STT_LANG_MAP.get(lang_code)
        if not mms_code:
            logger.warning(f"[MMS-STT] Unknown app language: {lang_code}")
            return None

        if not self._load_base():
            return None

        try:
            logger.info(f"[MMS-STT] Loading adapter for {mms_code}...")
            from copy import deepcopy
            proc = deepcopy(self._base_processor)
            proc.tokenizer.set_target_lang(mms_code)
            self._base_model.load_adapter(mms_code)
            self._models[lang_code] = (self._base_model, proc)
            logger.info(f"[MMS-STT] Adapter {mms_code} ready")
            return self._models[lang_code]
        except Exception as e:
            logger.error(f"[MMS-STT] FAILED to load adapter {mms_code}: {e}")
            return None

    def transcribe(self, audio_path: str, lang_code: str) -> Optional[str]:
        if not _MMS_STT_AVAILABLE:
            logger.warning("[MMS-STT] transcribe called but dependencies missing")
            return None
        if lang_code in MMS_STT_UNSUPPORTED:
            logger.info(f"[MMS-STT] {lang_code} not supported — returning None to trigger Khaya fallback")
            return None

        loaded = self._load_model(lang_code)
        if not loaded:
            logger.warning(f"[MMS-STT] No model loaded for lang={lang_code}")
            return None

        model, processor = loaded

        try:
            speech, sr = librosa.load(audio_path, sr=16000)
            if len(speech) == 0:
                logger.warning("[MMS-STT] Empty audio file")
                return None

            inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs).logits

            predicted_ids = torch.argmax(outputs, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            result = transcription.strip()
            logger.info(f"[MMS-STT] Transcribed ({lang_code}): {result[:80]}")
            return result if result else None

        except Exception as e:
            logger.error(f"[MMS-STT] Transcription failed: {e}")
            return None

    def health(self) -> dict:
        return {
            "available": _MMS_STT_AVAILABLE,
            "device": self._device,
            "base_loaded": self._base_model_loaded,
            "loaded_adapters": list(self._models.keys()),
            "supported_languages": list(MMS_STT_LANG_MAP.keys()),
            "unsupported_languages": list(MMS_STT_UNSUPPORTED),
        }


mms_stt = MMSSTTService()