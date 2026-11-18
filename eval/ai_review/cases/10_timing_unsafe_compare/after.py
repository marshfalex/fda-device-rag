import hashlib
import hmac

WEBHOOK_SECRET = b""  # loaded from env at startup


def verify(body: bytes, signature: str) -> bool:
    expected = hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
    return expected == signature
