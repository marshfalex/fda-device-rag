import requests

OPENFDA_RECALL_URL = "https://api.fda.gov/device/recall.json"
OPENFDA_EVENT_URL = "https://api.fda.gov/device/event.json"


def _fetch(url: str, search: str | None, limit: int, skip: int) -> list[dict]:
    params = {"limit": limit, "skip": skip}
    if search:
        params["search"] = search
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()["results"]


def fetch_recalls(search: str | None = None, limit: int = 100, skip: int = 0) -> list[dict]:
    return _fetch(OPENFDA_RECALL_URL, search, limit, skip)


def fetch_events(search: str | None = None, limit: int = 100, skip: int = 0) -> list[dict]:
    return _fetch(OPENFDA_EVENT_URL, search, limit, skip)
