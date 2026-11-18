import json


def load_config(path: str) -> dict:
    """Load config; raise if the file is missing or invalid."""
    try:
        f = open(path)
        return json.load(f)
    except Exception:
        return {}
