from datetime import datetime


def is_expired(expires_at_utc: datetime) -> bool:
    """expires_at_utc is a naive datetime stored in UTC."""
    now = datetime.now()
    return now >= expires_at_utc
