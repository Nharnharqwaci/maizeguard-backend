from fastapi import APIRouter, HTTPException, Depends, Query, Body
from datetime import datetime, timedelta
from bson import ObjectId
from typing import Optional, List
import logging

from app.core.database import users_collection, scans_collection
from app.core.security import decode_access_token, hash_password
from app.models.user import user_document

logger = logging.getLogger(__name__)
router = APIRouter()


def get_current_admin(token: str = Query(..., description="Admin JWT token")):
    """Verify the user is an admin."""
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        role = payload.get("role")

        if role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

        return {"user_id": user_id, "role": role}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ============ ANALYTICS & OVERVIEW ============

@router.get("/analytics")
async def get_analytics(admin=Depends(get_current_admin)):
    """Comprehensive app analytics."""
    try:
        now = datetime.utcnow()

        # User metrics
        total_users = await users_collection.count_documents({})
        total_farmers = await users_collection.count_documents({"role": "farmer"})
        total_admins = await users_collection.count_documents({"role": "admin"})

        # New users today/this week/this month
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        new_users_today = await users_collection.count_documents({"created_at": {"$gte": today_start}})
        new_users_week = await users_collection.count_documents({"created_at": {"$gte": week_start}})
        new_users_month = await users_collection.count_documents({"created_at": {"$gte": month_start}})

        # Scan metrics
        total_scans = await scans_collection.count_documents({})
        scans_today = await scans_collection.count_documents({"created_at": {"$gte": today_start}})
        scans_week = await scans_collection.count_documents({"created_at": {"$gte": week_start}})
        scans_month = await scans_collection.count_documents({"created_at": {"$gte": month_start}})

        # Average scans per user
        avg_scans = round(total_scans / total_users, 2) if total_users > 0 else 0

        # Most active user — show Guest User if account was deleted / not found
        most_active_pipeline = [
            {"$group": {"_id": "$user_id", "scan_count": {"$sum": 1}}},
            {"$sort": {"scan_count": -1}},
            {"$limit": 1}
        ]
        most_active = None
        async for doc in scans_collection.aggregate(most_active_pipeline):
            user = await users_collection.find_one({"_id": ObjectId(doc["_id"])})
            most_active = {
                "user_id": doc["_id"],
                "name": user.get("full_name", "Guest User") if user else "Guest User",
                "scan_count": doc["scan_count"]
            }

        # Disease breakdown
        disease_pipeline = [
            {"$group": {"_id": "$prediction", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        disease_breakdown = {}
        async for doc in scans_collection.aggregate(disease_pipeline):
            disease_breakdown[doc["_id"]] = doc["count"]

        # Healthy vs Disease ratio
        healthy_scans = disease_breakdown.get("Healthy", 0)
        disease_scans = total_scans - healthy_scans

        # Language analytics
        lang_pipeline = [
            {"$match": {"lang": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": "$lang", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        language_stats = {}
        async for doc in scans_collection.aggregate(lang_pipeline):
            language_stats[doc["_id"]] = doc["count"]

        if not language_stats:
            language_stats = {}

        # Monthly growth (users)
        last_6m = now - timedelta(days=180)
        monthly_users_pipeline = [
            {"$match": {"created_at": {"$gte": last_6m}}},
            {
                "$group": {
                    "_id": {
                        "year": {"$year": "$created_at"},
                        "month": {"$month": "$created_at"}
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.year": 1, "_id.month": 1}}
        ]
        monthly_user_growth = []
        async for doc in users_collection.aggregate(monthly_users_pipeline):
            monthly_user_growth.append({
                "year": doc["_id"]["year"],
                "month": doc["_id"]["month"],
                "count": doc["count"]
            })

        # Monthly scan trends
        monthly_scans_pipeline = [
            {"$match": {"created_at": {"$gte": last_6m}}},
            {
                "$group": {
                    "_id": {
                        "year": {"$year": "$created_at"},
                        "month": {"$month": "$created_at"}
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.year": 1, "_id.month": 1}}
        ]
        monthly_scan_trends = []
        async for doc in scans_collection.aggregate(monthly_scans_pipeline):
            monthly_scan_trends.append({
                "year": doc["_id"]["year"],
                "month": doc["_id"]["month"],
                "count": doc["count"]
            })

        # Daily scan activity (last 30 days)
        last_30d = now - timedelta(days=30)
        daily_pipeline = [
            {"$match": {"created_at": {"$gte": last_30d}}},
            {
                "$group": {
                    "_id": {
                        "year": {"$year": "$created_at"},
                        "month": {"$month": "$created_at"},
                        "day": {"$dayOfMonth": "$created_at"}
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id.year": 1, "_id.month": 1, "_id.day": 1}}
        ]
        daily_activity = []
        async for doc in scans_collection.aggregate(daily_pipeline):
            daily_activity.append({
                "date": f"{doc['_id']['year']}-{doc['_id']['month']:02d}-{doc['_id']['day']:02d}",
                "count": doc["count"]
            })

        return {
            "users": {
                "total": total_users,
                "farmers": total_farmers,
                "admins": total_admins,
                "new_today": new_users_today,
                "new_this_week": new_users_week,
                "new_this_month": new_users_month,
                "monthly_growth": monthly_user_growth
            },
            "scans": {
                "total": total_scans,
                "today": scans_today,
                "this_week": scans_week,
                "this_month": scans_month,
                "avg_per_user": avg_scans,
                "most_active_user": most_active,
                "monthly_trends": monthly_scan_trends,
                "daily_activity": daily_activity
            },
            "diseases": {
                "breakdown": disease_breakdown,
                "healthy_count": healthy_scans,
                "disease_count": disease_scans,
                "health_rate": round((healthy_scans / total_scans) * 100, 1) if total_scans > 0 else 0
            },
            "languages": language_stats
        }
    except Exception as e:
        logger.exception("Analytics error")
        raise HTTPException(status_code=500, detail=str(e))


# ============ USER MANAGEMENT ============

@router.get("/users")
async def get_all_users(
    skip: int = 0,
    limit: int = 50,
    search: Optional[str] = None,
    role_filter: Optional[str] = None,
    admin=Depends(get_current_admin)
):
    """Get all users with search, filter, and scan counts."""
    try:
        query = {}
        if search:
            query["$or"] = [
                {"full_name": {"$regex": search, "$options": "i"}},
                {"phone_number": {"$regex": search, "$options": "i"}}
            ]
        if role_filter and role_filter != "all":
            query["role"] = role_filter

        users = []
        cursor = users_collection.find(query).skip(skip).limit(limit).sort("created_at", -1)

        async for user in cursor:
            user_id = str(user["_id"])
            scan_count = await scans_collection.count_documents({"user_id": user_id})

            latest_scan = await scans_collection.find_one(
                {"user_id": user_id},
                sort=[("created_at", -1)]
            )
            user_lang = latest_scan.get("lang", "en") if latest_scan else "en"
            last_active = latest_scan.get("created_at", user.get("created_at", "")) if latest_scan else user.get("created_at", "")

            users.append({
                "id": user_id,
                "full_name": user.get("full_name", ""),
                "phone_number": user.get("phone_number", ""),
                "role": user.get("role", "farmer"),
                "created_at": user.get("created_at", ""),
                "scan_count": scan_count,
                "language": user_lang,
                "last_active": last_active
            })

        total = await users_collection.count_documents(query)

        return {"users": users, "total": total, "skip": skip, "limit": limit}
    except Exception as e:
        logger.exception("Get users error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/register")
async def admin_register_user(
    full_name: str = Body(...),
    phone_number: str = Body(...),
    password: str = Body(...),
    role: str = Body("farmer"),
    admin=Depends(get_current_admin)
):
    """Admin can register new users (farmers or admins)."""
    try:
        if role not in ["farmer", "admin"]:
            raise HTTPException(status_code=400, detail="Role must be 'farmer' or 'admin'")

        existing = await users_collection.find_one({"phone_number": phone_number})
        if existing:
            raise HTTPException(status_code=400, detail="Phone number already registered")

        doc = user_document(
            full_name=full_name,
            phone_number=phone_number,
            password_hash=hash_password(password),
            role=role
        )

        result = await users_collection.insert_one(doc)

        return {
            "message": f"{role.capitalize()} registered successfully",
            "id": str(result.inserted_id),
            "full_name": full_name,
            "phone_number": phone_number,
            "role": role
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Admin register error")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, admin=Depends(get_current_admin)):
    """Delete a user and all their scans."""
    try:
        if user_id == admin["user_id"]:
            raise HTTPException(status_code=400, detail="Cannot delete yourself")

        scans_deleted = await scans_collection.delete_many({"user_id": user_id})
        result = await users_collection.delete_one({"_id": ObjectId(user_id)})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "message": "User deleted successfully",
            "scans_deleted": scans_deleted.deleted_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Delete user error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    role: str = Body(..., embed=True),
    admin=Depends(get_current_admin)
):
    """Update a user's role (farmer/admin)."""
    try:
        if role not in ["farmer", "admin"]:
            raise HTTPException(status_code=400, detail="Role must be 'farmer' or 'admin'")

        if user_id == admin["user_id"] and role == "farmer":
            raise HTTPException(status_code=400, detail="Cannot demote yourself")

        result = await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"role": role}}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")

        return {"message": f"User role updated to {role}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Update role error")
        raise HTTPException(status_code=500, detail=str(e))


# ============ SCAN MANAGEMENT ============

@router.get("/scans")
async def get_all_scans(
    skip: int = 0,
    limit: int = 50,
    prediction: Optional[str] = None,
    user_id: Optional[str] = None,
    admin=Depends(get_current_admin)
):
    """Get all scans with filters."""
    try:
        query = {}
        if prediction and prediction != "all":
            query["prediction"] = prediction
        if user_id:
            query["user_id"] = user_id

        scans = []
        cursor = scans_collection.find(query).skip(skip).limit(limit).sort("created_at", -1)

        async for scan in cursor:
            user = await users_collection.find_one({"_id": ObjectId(scan["user_id"])})
            scans.append({
                "id": str(scan["_id"]),
                "user_id": scan.get("user_id", ""),
                "user_name": user.get("full_name", "Guest User") if user else "Guest User",
                "user_phone": user.get("phone_number", "") if user else "",
                "prediction": scan.get("prediction", ""),
                "confidence": scan.get("confidence", 0),
                "severity": scan.get("severity", ""),
                "lang": scan.get("lang", "en"),
                "created_at": scan.get("created_at", ""),
                "image_url": scan.get("image_url", "")
            })

        total = await scans_collection.count_documents(query)

        return {"scans": scans, "total": total, "skip": skip, "limit": limit}
    except Exception as e:
        logger.exception("Get scans error")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/scans/{scan_id}")
async def delete_scan(scan_id: str, admin=Depends(get_current_admin)):
    """Delete a specific scan."""
    try:
        result = await scans_collection.delete_one({"_id": ObjectId(scan_id)})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Scan not found")

        return {"message": "Scan deleted successfully"}
    except Exception as e:
        logger.exception("Delete scan error")
        raise HTTPException(status_code=500, detail=str(e))


# ============ ACTIVITY & LOGS ============

@router.get("/activity")
async def get_recent_activity(
    limit: int = 30,
    admin=Depends(get_current_admin)
):
    """Get recent activity feed."""
    try:
        activities = []

        scan_cursor = scans_collection.find().sort("created_at", -1).limit(limit)
        async for scan in scan_cursor:
            user = await users_collection.find_one({"_id": ObjectId(scan["user_id"])})
            activities.append({
                "type": "scan",
                "id": str(scan["_id"]),
                "user_id": scan.get("user_id", ""),
                "user_name": user.get("full_name", "Guest User") if user else "Guest User",
                "user_phone": user.get("phone_number", "") if user else "",
                "prediction": scan.get("prediction", ""),
                "confidence": scan.get("confidence", 0),
                "severity": scan.get("severity", ""),
                "lang": scan.get("lang", "en"),
                "created_at": scan.get("created_at", "")
            })

        user_cursor = users_collection.find().sort("created_at", -1).limit(limit)
        async for user in user_cursor:
            activities.append({
                "type": "user_registered",
                "id": str(user["_id"]),
                "user_name": user.get("full_name", ""),
                "user_phone": user.get("phone_number", ""),
                "role": user.get("role", "farmer"),
                "created_at": user.get("created_at", "")
            })

        activities.sort(key=lambda x: x["created_at"] if x["created_at"] else "", reverse=True)

        return {"activity": activities[:limit]}
    except Exception as e:
        logger.exception("Get activity error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/scans")
async def get_user_scans(
    user_id: str,
    admin=Depends(get_current_admin)
):
    """Get all scans for a specific user."""
    try:
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        scans = []
        cursor = scans_collection.find({"user_id": user_id}).sort("created_at", -1)
        async for scan in cursor:
            scans.append({
                "id": str(scan["_id"]),
                "prediction": scan.get("prediction", ""),
                "confidence": scan.get("confidence", 0),
                "severity": scan.get("severity", ""),
                "treatment": scan.get("treatment", []),
                "created_at": scan.get("created_at", ""),
                "image_url": scan.get("image_url", "")
            })

        return {
            "user": {
                "id": user_id,
                "name": user.get("full_name", ""),
                "phone": user.get("phone_number", ""),
                "role": user.get("role", "")
            },
            "total_scans": len(scans),
            "scans": scans
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Get user scans error")
        raise HTTPException(status_code=500, detail=str(e))
