from datetime import datetime

def user_document(
    full_name: str,
    phone_number: str,
    password_hash: str,
    role: str = "farmer"
) -> dict:
    return {
        "full_name": full_name,
        "phone_number": phone_number,
        "password_hash": password_hash,
        "role": role,
        "created_at": datetime.utcnow()
    }