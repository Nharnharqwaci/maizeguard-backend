from fastapi import APIRouter, HTTPException
from app.schemas.register_schema import RegisterRequest
from app.schemas.login_schema import LoginRequest
from app.models.user import user_document
from app.core.database import users_collection
from app.core.security import hash_password, verify_password, create_access_token
from bson import ObjectId
from datetime import datetime

router = APIRouter()


@router.post("/register")
async def register(user: RegisterRequest):
    # check if phone number already exists
    existing = await users_collection.find_one(
        {"phone_number": user.phone_number}
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Phone number already registered"
        )

    doc = user_document(
        full_name=user.full_name,
        phone_number=user.phone_number,
        password_hash=hash_password(user.password),
        role="farmer"
    )
    result = await users_collection.insert_one(doc)

    return {
        "message": "Registration successful",
        "id": str(result.inserted_id)
    }


@router.post("/login")
async def login(user: LoginRequest):
    db_user = await users_collection.find_one(
        {"phone_number": user.phone_number}
    )
    if not db_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )
    if not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_access_token({
        "sub": str(db_user["_id"]),
        "role": db_user["role"]
    })

    return {
        "message": "Login successful",
        "access_token": token,
        "token_type": "bearer",
        "user_id": str(db_user["_id"]),
        "role": db_user["role"],
        "full_name": db_user["full_name"]
    }