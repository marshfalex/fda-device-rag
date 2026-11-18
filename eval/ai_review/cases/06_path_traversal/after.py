from pathlib import Path

UPLOAD_DIR = Path("/srv/uploads")


def read_upload(filename: str) -> bytes:
    path = UPLOAD_DIR / filename
    return path.read_bytes()
