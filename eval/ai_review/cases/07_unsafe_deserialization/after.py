import base64
import pickle


def load_session(cookie_value: str) -> dict:
    raw = base64.b64decode(cookie_value)
    return pickle.loads(raw)
