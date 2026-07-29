from pathlib import Path

import requests

# Some hosts (e.g. fda.gov) run bot-detection that silently 404s the default
# python-requests User-Agent, redirecting to an "apology" page instead of the
# actual file. A normal browser-like User-Agent avoids this.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def _derive_filename(url: str) -> str:
    segments = [s for s in url.split("/") if s]
    filename = segments[-1] if segments else "download"
    # Some hosts (e.g. fda.gov/media/<id>/download) serve every document at a
    # URL ending literally in "download" -- fall back to the preceding
    # segment (the id) so distinct documents don't collide on one filename.
    if filename.lower() == "download" and len(segments) >= 2:
        filename = segments[-2]
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    return filename


def download_pdfs(url_list: list[str], dest_dir: Path) -> list[Path]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for url in url_list:
        filename = _derive_filename(url)
        dest_path = dest_dir / filename

        response = requests.get(url, headers=_REQUEST_HEADERS, timeout=60)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        downloaded.append(dest_path)

    return downloaded
