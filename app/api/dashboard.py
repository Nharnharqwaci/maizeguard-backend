from fastapi import APIRouter, HTTPException
from app.core.database import scans_collection
from bson import ObjectId
from datetime import datetime, timezone

router = APIRouter()


def format_scan(scan: dict) -> dict:
    return {
        "_id":        str(scan["_id"]),
        "prediction": scan.get("prediction", "Unknown"),
        "confidence": scan.get("confidence", 0),
        "severity":   scan.get("severity", "low"),
        "image_url":  scan.get("image_url", ""),
        "created_at": scan["created_at"].isoformat()
        if isinstance(scan.get("created_at"), datetime)
        else str(scan.get("created_at", "")),
    }


@router.get("/stats/{user_id}")
async def get_dashboard_stats(user_id: str):
    try:
        # validate user_id format
        try:
            ObjectId(user_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid user ID")

        # all scans for this user
        all_scans = await scans_collection.find(
            {"user_id": user_id}
        ).sort("created_at", -1).to_list(length=1000)

        total_scans   = len(all_scans)
        disease_cases = sum(
            1 for s in all_scans
            if s.get("prediction") not in ("Healthy", "Uncertain")
        )
        healthy_count = sum(
            1 for s in all_scans
            if s.get("prediction") == "Healthy"
        )

        # disease breakdown
        disease_breakdown: dict[str, int] = {}
        for scan in all_scans:
            pred = scan.get("prediction", "Unknown")
            disease_breakdown[pred] = disease_breakdown.get(pred, 0) + 1

        # 5 most recent scans
        recent_scans = [format_scan(s) for s in all_scans[:5]]

        # last scan timestamp
        last_scan = None
        if all_scans:
            raw = all_scans[0].get("created_at")
            if isinstance(raw, datetime):
                last_scan = raw.isoformat()

        return {
            "total_scans":        total_scans,
            "disease_cases":      disease_cases,
            "healthy_count":      healthy_count,
            "disease_breakdown":  disease_breakdown,
            "recent_scans":       recent_scans,
            "last_scan":          last_scan,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{user_id}")
async def get_scan_history(user_id: str):
    try:
        ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID")

    scans = await scans_collection.find(
        {"user_id": user_id}
    ).sort("created_at", -1).to_list(length=100)

    return [format_scan(s) for s in scans]