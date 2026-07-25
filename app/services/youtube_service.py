import requests
from app.core.config import YOUTUBE_API_KEY

SEARCHES = {
    "Healthy":              "healthy maize farming Ghana OR Africa",
    "MSV":                  "maize streak virus control Ghana OR Africa",
    "Common_Rust":          "common rust maize management Ghana OR Africa",
    "Gray_Leaf_Spot":       "gray leaf spot maize control Ghana OR Africa",
    "Northern_Leaf_Blight": "northern leaf blight maize Ghana OR Africa management",
    "Southern_Leaf_Blight": "southern leaf blight maize Ghana OR Africa  treatment",
    "Uncertain":            None,
}


def get_videos(prediction: str) -> list[dict]:
    query = SEARCHES.get(prediction)
    if not query:
        return []

    try:
        # Step 1 — search for videos
        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            "part":              "snippet",
            "q":                 query,
            "type":              "video",
            "maxResults":        15,          # fetch more so we can filter
            "regionCode":        "GH",
            "relevanceLanguage": "en",
            "videoEmbeddable":   "true",     # only embeddable videos
            "safeSearch":        "strict",
            "order":             "relevance",
            "key":               YOUTUBE_API_KEY,
        }

        search_resp = requests.get(
            search_url,
            params=search_params,
            timeout=10
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()

        video_ids = [
            item["id"]["videoId"]
            for item in search_data.get("items", [])
            if item.get("id", {}).get("videoId")
        ]

        if not video_ids:
            return []

        # Step 2 — verify each video is actually embeddable
        # using the videos endpoint with status part
        videos_url = "https://www.googleapis.com/youtube/v3/videos"
        videos_params = {
            "part": "snippet,status,contentDetails",
            "id":   ",".join(video_ids),
            "key":  YOUTUBE_API_KEY,
        }

        videos_resp = requests.get(
            videos_url,
            params=videos_params,
            timeout=10
        )
        videos_resp.raise_for_status()
        videos_data = videos_resp.json()

        results = []
        for item in videos_data.get("items", []):
            status = item.get("status", {})

            # skip videos that are not embeddable
            if not status.get("embeddable", False):
                continue

            # skip private or unlisted videos
            if status.get("privacyStatus") not in ("public",):
                continue

            snippet   = item["snippet"]
            video_id  = item["id"]
            thumbnail = (
                snippet.get("thumbnails", {})
                .get("high", {})
                .get("url", "")
                or snippet.get("thumbnails", {})
                .get("medium", {})
                .get("url", "")
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

        return results

    except Exception as e:
        print(f"YouTube API Error for '{prediction}': {e}")
        return []