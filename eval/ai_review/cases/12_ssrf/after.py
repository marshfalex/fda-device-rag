import requests


def fetch_preview(url: str) -> str:
    """Fetch a user-supplied URL to build a link preview."""
    resp = requests.get(url, timeout=5)
    return resp.text[:500]
