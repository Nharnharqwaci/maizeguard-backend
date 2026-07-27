import os
import logging
import requests
import json

logger = logging.getLogger(__name__)

ARKESEL_API_KEY = os.getenv("ARKESEL_API_KEY", "")
ARKESEL_SENDER_ID = os.getenv("ARKESEL_SENDER_ID", "MaizeAI")


def normalize_ghana_number(phone_number: str) -> str:
    """Normalize to E.164 +233XXXXXXXXX format."""
    cleaned = phone_number.strip().replace(" ", "").replace("-", "").replace("+", "")

    if cleaned.startswith("0"):
        cleaned = "233" + cleaned[1:]

    if not cleaned.startswith("233") and len(cleaned) == 9:
        cleaned = "233" + cleaned

    # Ensure leading + for E.164
    if not cleaned.startswith("+"):
        cleaned = "+" + cleaned

    return cleaned


def send_sms_arkesel(phone_number: str, message: str) -> dict:
    """
    Send SMS via Arkesel Ghana.
    Returns detailed result for debugging.
    """
    result = {
        "success": False,
        "attempts": [],
        "final_error": None,
        "api_key_set": bool(ARKESEL_API_KEY),
        "sender_id_set": bool(ARKESEL_SENDER_ID),
        "sender_id_value": ARKESEL_SENDER_ID,
    }

    if not ARKESEL_API_KEY:
        result["final_error"] = "ARKESEL_API_KEY not set in environment"
        logger.error(result["final_error"])
        return result

    cleaned = normalize_ghana_number(phone_number)
    logger.info(f"[Arkesel] Normalized number: {cleaned} (from: {phone_number})")

    url = "https://sms.arkesel.com/api/v2/sms/send"
    headers = {
        "Content-Type": "application/json",
        "api-key": ARKESEL_API_KEY,
    }
    payload = {
        "sender": ARKESEL_SENDER_ID,
        "message": message,
        "recipients": [cleaned],
    }

    logger.info(f"[Arkesel] URL: {url}")
    logger.info(f"[Arkesel] Payload: {json.dumps(payload)}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)

        logger.info(f"[Arkesel] HTTP Status: {response.status_code}")
        logger.info(f"[Arkesel] Raw Response: {response.text}")

        try:
            data = response.json()
        except Exception as e:
            result["attempts"].append({
                "format": "json",
                "http_status": response.status_code,
                "error": f"Invalid JSON response: {str(e)}",
                "raw": response.text[:500]
            })
            result["final_error"] = f"Arkesel returned non-JSON: {response.text[:200]}"
            return result

        result["attempts"].append({
            "format": "json",
            "http_status": response.status_code,
            "response": data
        })

        # Check Arkesel response
        status = data.get("status", "")
        msg = data.get("message", "")
        response_data = data.get("data", {})

        logger.info(f"[Arkesel] Parsed -> status={status}, message={msg}, data={response_data}")

        if status == "success":
            result["success"] = True
            result["arkesel_status"] = status
            result["arkesel_message"] = msg
            result["arkesel_data"] = response_data
            logger.info("[Arkesel] SMS reported as sent successfully")
        else:
            result["final_error"] = f"Arkesel rejected: status={status}, message={msg}"
            result["arkesel_status"] = status
            result["arkesel_message"] = msg
            logger.error(f"[Arkesel] {result['final_error']}")

        return result

    except requests.exceptions.Timeout:
        result["final_error"] = "Request timed out after 20s"
        logger.error(f"[Arkesel] {result['final_error']}")
        return result
    except requests.exceptions.ConnectionError as e:
        result["final_error"] = f"Connection error: {str(e)}"
        logger.error(f"[Arkesel] {result['final_error']}")
        return result
    except Exception as e:
        result["final_error"] = f"Unexpected error: {str(e)}"
        logger.error(f"[Arkesel] {result['final_error']}")
        return result
