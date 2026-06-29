from fastapi import APIRouter
from app.core.database import (
    scans_collection
)

router = APIRouter()


@router.get("/stats/{user_id}")
async def get_stats(user_id: str):

    total_scans = await (
        scans_collection.count_documents(
            {
            "user_id": user_id
            }
        )
    )

    disease_cases = (
        scans_collection.count_documents(
            {
                "user_id": user_id,
                "prediction": 'MSV'
            }
        )
    )

    recent_scans = list(
        scans_collection
        .find({
            "user_id": user_id
        })
        .sort(
            "created_at",
            -1
        )
        .limit(10)
    )

    for scan in recent_scans:

        scan["_id"] = str(
            scan["_id"]
        )

        if "created_at" in scan:
            scan["created_at"] = str(
                scan["created_at"]
            )

    return {
        "total_scans": total_scans,
        "disease_cases": disease_cases,
        "recent_scans": recent_scans
    }