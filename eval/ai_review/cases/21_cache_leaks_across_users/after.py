from functools import lru_cache

from flask_login import current_user


@lru_cache(maxsize=1024)
def dashboard_data(page: int) -> dict:
    """Expensive per-user dashboard query, cached for speed."""
    return {"user": current_user.id, "items": current_user.items(page=page)}
