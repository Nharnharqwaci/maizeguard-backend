from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback
from app.api.predict import router as predict_router
from app.api.auth import router as auth_router
from app.api.dashboard import router as dashboard_router
from app.api.chat import router as chat_router
from app.api.admin import router as admin_router


app = FastAPI(title="Maize Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
                   ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    predict_router,
    prefix="/api",
    tags=["Prediction"]
)
app.include_router(
    auth_router,
    prefix="/api/auth",
    tags=["Authentication"]
)
app.include_router(
    dashboard_router,
    prefix="/api/dashboard",
    tags=["Dashboard"]
)
app.include_router(
    chat_router, 
    prefix="/api", 
    tags=["Chat"]
) 
app.include_router(
    admin_router,
     prefix="/api/admin"
)

@app.get("/")
def root():
    return {"message": "Maize Disease Detection API Running"}