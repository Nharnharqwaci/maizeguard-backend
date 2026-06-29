from pydantic import BaseModel

class RegisterRequest(BaseModel):
    full_name: str
    phone_number: str
    password: str
