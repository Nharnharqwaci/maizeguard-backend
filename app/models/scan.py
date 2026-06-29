from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Scan(BaseModel):
    user_id: str
    image_url: str
    prediction: str
    confidence: float
    treatment: List[str]
    created_at: datetime = datetime.utcnow()