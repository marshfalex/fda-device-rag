import hashlib
import hmac
import os

WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"].encode()


def verify(body: bytes, signature: str) -> bool:
    expected = hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
