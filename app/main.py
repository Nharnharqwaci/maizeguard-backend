# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.predict import router as predict_router
from app.api.auth import router as auth_router
from app.api.dashboard import router as dashboard_router
from app.api.chat import router as chat_router
from app.api.admin import router as admin_router

from app.services.mms_tts_service import mms_tts
from app.services.mms_stt_service import mms_stt


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload Meta MMS models on startup."""
    # TTS: Twi (aka) and English (eng) only — Dagbani has no MMS model
    mms_tts.preload(["en", "tw"])
    # STT: Twi (aka) and English (eng) only — Dagbani has no MMS adapter
    mms_stt.preload(["en", "tw"])
    yield


app = FastAPI(
    title="Maize Disease API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://maizeguard-frontend-8772.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api", tags=["Prediction"])
app.include_router(auth_router,    prefix="/api/auth", tags=["Authentication"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(chat_router,    prefix="/api", tags=["Chat"])
app.include_router(admin_router,   prefix="/api/admin", tags=["Admin"])


@app.get("/")
def root():
    return {"message": "Maize Disease Detection API Running"}