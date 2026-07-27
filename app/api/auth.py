import os
from fastapi import APIRouter, HTTPException, Body
from app.schemas.register_schema import RegisterRequest
from app.schemas.login_schema import LoginRequest
from app.models.user import user_document
from app.core.database import users_collection
from app.core.security import hash_password, verify_password, create_access_token
from app.services.arkesel_service import send_sms_arkesel
from bson import ObjectId
from datetime import datetime
import secrets
import string
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/register")
async def register(user: RegisterRequest):
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

    requires_password_change = db_user.get("requires_password_change", False)

    return {
        "message": "Login successful",
        "access_token": token,
        "token_type": "bearer",
        "user_id": str(db_user["_id"]),
        "role": db_user["role"],
        "full_name": db_user["full_name"],
        "requires_password_change": requires_password_change
    }


@router.post("/forgot-password")
async def forgot_password(phone_number: str = Body(..., embed=True)):
    """Generate a temporary password and send it via SMS using Arkesel."""
    try:
        user = await users_collection.find_one({"phone_number": phone_number})
        if not user:
            raise HTTPException(
                status_code=404,
                detail="No account found with this phone number"
            )

        alphabet = string.ascii_letters + string.digits
        temp_password = ''.join(secrets.choice(alphabet) for _ in range(10))
        hashed = hash_password(temp_password)

        await users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {
                "password_hash": hashed,
                "requires_password_change": True
            }}
        )

        user_name = user.get("full_name", "User")
        sms_message = f"Hi {user_name}, your MaizeAI temporary password is: {temp_password}. Log in and change it immediately."

        logger.info(f"[ForgotPassword] Sending SMS to {phone_number}")
        sms_result = send_sms_arkesel(phone_number, sms_message)
        logger.info(f"[ForgotPassword] SMS result: {sms_result}")

        # Return full debug info so frontend can show details
        return {
            "message": "If this phone number is registered, a temporary password has been sent via SMS.",
            "sms_delivered": sms_result.get("success", False),
            "sms_debug": sms_result,
            "phone_normalized": sms_result.get("attempts", [{}])[0].get("format", "") if sms_result.get("attempts") else ""
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Forgot password error")
        raise HTTPException(status_code=500, detail="Password reset failed")


@router.post("/change-password")
async def change_password(
    current_password: str = Body(...),
    new_password: str = Body(...),
    token: str = Body(...)
):
    """Allow authenticated users to change their password."""
    from app.core.security import decode_access_token
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")

        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not verify_password(current_password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect")

        await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "password_hash": hash_password(new_password),
                "requires_password_change": False
            }}
        )

        return {"message": "Password changed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Change password error")
        raise HTTPException(status_code=500, detail="Failed to change password")


@router.post("/test-sms")
async def test_sms(
    phone_number: str = Body(...),
    message: str = Body("Test from MaizeAI")
):
    """Debug endpoint to test Arkesel SMS directly."""
    try:
        result = send_sms_arkesel(phone_number, message)
        return {
            "test_result": result,
            "env_check": {
                "api_key_configured": bool(os.getenv("ARKESEL_API_KEY")),
                "sender_id_configured": os.getenv("ARKESEL_SENDER_ID", "NOT_SET"),
            }
        }
    except Exception as e:
        logger.exception("Test SMS error")
        raise HTTPException(status_code=500, detail=str(e))
