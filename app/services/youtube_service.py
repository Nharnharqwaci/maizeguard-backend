"""
youtube_service.py — Fetches relevant agriculture videos.
PRIMARY: YouTube Data API v3 (requires YOUTUBE_API_KEY env var).
FALLBACK: Curated static videos (works offline / without API key).
"""

import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# ── CURATED FALLBACK VIDEOS ──
# videoId MUST be the 11-character YouTube ID.
# Example: https://youtu.be/h7_tXBCcVaw?si=XXXX → videoId = "h7_tXBCcVaw"
FALLBACK_VIDEOS: Dict[str, List[Dict]] = {
    "Common_Rust": [
        {
            "videoId": "h7_tXBCcVaw",
            "title": "Maize Common Rust: Identification & Control in Ghana",
            "channel": "MOFA Extension",
            "thumbnail": "https://img.youtube.com/vi/h7_tXBCcVaw/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=h7_tXBCcVaw",
        },
        {
            "videoId": "dFOMc-rSK7Q",
            "title": "Fungicide Application for Rust in Maize",
            "channel": "CABI Agriculture",
            "thumbnail": "https://img.youtube.com/vi/dFOMc-rSK7Q/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=dFOMc-rSK7Q",
        },
    ],
    "Gray_Leaf_Spot": [
        {
            "videoId": "SAp80XsXrJ4",
            "title": "Gray Leaf Spot Management in Maize",
            "channel": "SyngentaAgUS",
            "thumbnail": "https://img.youtube.com/vi/SAp80XsXrJ4/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=SAp80XsXrJ4",
        },
        {
            "videoId": "KlJZ94RPC4I",
            "title": "Integrated Pest Management for Gray Leaf Spot on Field Corn",
            "channel": "Cornell Integrated Pest Management",
            "thumbnail": "https://img.youtube.com/vi/KlJZ94RPC4I/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=KlJZ94RPC4I",
        },
    ],
    "Healthy": [
        {
            "videoId": "5jQSKDWU9Zs",
            "title": "Best Practices for Healthy Maize Farming in Ghana",
            "channel": "Debi Naa - Expert Africa",
            "thumbnail": "https://img.youtube.com/vi/5jQSKDWU9Zs/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=5jQSKDWU9Zs",
        },
        {
            "videoId": "90aapJnrAIM",
            "title": "Unlocking the Lucrative Potential of Maize Farming in Ghana",
            "channel": "The Ghanaian Farmer TV",
            "thumbnail": "https://img.youtube.com/vi/90aapJnrAIM/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=90aapJnrAIM",
        },
    ],
    "MSV": [
        {
            "videoId": "GS6q-D71NgE",
            "title": "MAIZE STREAK VIRUS MODE OF INFECTION AND CONTROL",
            "channel": "Theresa Feka Farms",
            "thumbnail": "https://img.youtube.com/vi/GS6q-D71NgE/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=GS6q-D71NgE",
        },
        {
            "videoId": "ohqEWgzKnXo",
            "title": "Maize Strick virus",
            "channel": "THE GADENAZ",
            "thumbnail": "https://img.youtube.com/vi/ohqEWgzKnXo/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=ohqEWgzKnXo",
        },
    ],
    "Northern_Leaf_Blight": [
        {
            "videoId": "uafRy5EqwBQ",
            "title": "Northern Corn Leaf Blight Management",
            "channel": "Pioneer Seeds United States",
            "thumbnail": "https://img.youtube.com/vi/uafRy5EqwBQ/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=uafRy5EqwBQ",
        },
        {
            "videoId": "hF1XZcouNcg",
            "title": "How to Achieve a Million Tons of Corn on a piece of Land",
            "channel": "Rising Farmers",
            "thumbnail": "https://img.youtube.com/vi/hF1XZcouNcg/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=hF1XZcouNcg",
        },
    ],
    "Southern_Leaf_Blight": [
        {
            "videoId": "oAI6o__Ccsk",
            "title": "Southern Leaf Blight",
            "channel": "wikipedia tts",
            "thumbnail": "https://img.youtube.com/vi/oAI6o__Ccsk/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=oAI6o__Ccsk",
        },
        {
            "videoId": "NiBXpz_RWo4",
            "title": "Maize Disease in Plant Pathology",
            "channel": "African Lens Extra",
            "thumbnail": "https://img.youtube.com/vi/NiBXpz_RWo4/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=NiBXpz_RWo4",
        },
    ],
    "Uncertain": [
        {
        },
    ],
}

SEARCHES = {
    "Healthy":              "healthy maize farming Ghana OR Africa",
    "MSV":                  "maize streak virus control Ghana OR Africa",
    "Common_Rust":          "common rust maize management Ghana OR Africa",
    "Gray_Leaf_Spot":       "gray leaf spot maize control Ghana OR Africa",
    "Northern_Leaf_Blight": "northern leaf blight maize Ghana OR Africa management",
    "Southern_Leaf_Blight": "southern leaf blight maize Ghana OR Africa treatment",
    "Uncertain":            None,
}


def _search_youtube_api(prediction: str, lang: str = "en") -> List[Dict]:
    """Call YouTube Data API v3. Returns [] on any failure."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.info("[YouTube] YOUTUBE_API_KEY not set — skipping API search")
        return []

    try:
        import requests

        query = SEARCHES.get(prediction)
        if not query:
            return []

        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            "part":              "snippet",
            "q":                 query,
            "type":              "video",
            "maxResults":        15,
            "regionCode":        "GH",
            "relevanceLanguage": "en",
            "videoEmbeddable":   "true",
            "safeSearch":        "strict",
            "order":             "relevance",
            "key":               api_key,
        }

        search_resp = requests.get(search_url, params=search_params, timeout=10)
        search_resp.raise_for_status()
        search_data = search_resp.json()

        video_ids = [
            item["id"]["videoId"]
            for item in search_data.get("items", [])
            if item.get("id", {}).get("videoId")
        ]

        if not video_ids:
            logger.info(f"[YouTube] No video IDs found for {prediction}")
            return []

        videos_url = "https://www.googleapis.com/youtube/v3/videos"
        videos_params = {
            "part": "snippet,status,contentDetails",
            "id":   ",".join(video_ids),
            "key":  api_key,
        }

        videos_resp = requests.get(videos_url, params=videos_params, timeout=10)
        videos_resp.raise_for_status()
        videos_data = videos_resp.json()

        results = []
        for item in videos_data.get("items", []):
            status = item.get("status", {})

            if not status.get("embeddable", False):
                continue
            if status.get("privacyStatus") not in ("public",):
                continue

            snippet   = item["snippet"]
            video_id  = item["id"]
            thumbnail = (
                snippet.get("thumbnails", {}).get("high", {}).get("url", "")
                or snippet.get("thumbnails", {}).get("medium", {}).get("url", "")
            )

            results.append({
                "videoId":   video_id,
                "title":     snippet.get("title", ""),
                "channel":   snippet.get("channelTitle", ""),
                "thumbnail": thumbnail,
                "url":       f"https://www.youtube.com/watch?v={video_id}",
            })

            if len(results) == 4:
                break

        logger.info(f"[YouTube] API returned {len(results)} videos for {prediction}")
        return results

    except Exception as e:
        logger.error(f"[YouTube] API search failed for {prediction}: {e}")
        return []


def get_videos(prediction: str, lang: str = "en") -> List[Dict]:
    """
    Fetch relevant YouTube videos.
    1. Try live YouTube Data API (needs YOUTUBE_API_KEY).
    2. Fall back to curated static list (always works).
    """
    api_results = _search_youtube_api(prediction, lang)
    if api_results:
        return api_results

    fallback = FALLBACK_VIDEOS.get(prediction, FALLBACK_VIDEOS.get("Uncertain", []))
    logger.info(f"[YouTube] Using {len(fallback)} fallback videos for {prediction}")
    return fallback