from pathlib import Path

import requests


def download_pdfs(url_list: list[str], dest_dir: Path) -> list[Path]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for url in url_list:
        filename = url.rsplit("/", 1)[-1]
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        dest_path = dest_dir / filename

        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        downloaded.append(dest_path)

    return downloaded
