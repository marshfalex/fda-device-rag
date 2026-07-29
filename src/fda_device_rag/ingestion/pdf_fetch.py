from pathlib import Path

import requests


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

        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        downloaded.append(dest_path)

    return downloaded
