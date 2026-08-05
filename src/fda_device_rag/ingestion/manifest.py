import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ManifestEntry:
    source_type: str
    identifier: str
    url: str
    retrieved_date: str


class ManifestWriter:
    FIELDNAMES = ["source_type", "identifier", "url", "retrieved_date"]

    def __init__(self, path: Path):
        self._path = Path(path)
        is_new = not self._path.exists()
        self._file = open(self._path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if is_new:
            self._writer.writeheader()

    def write(self, entry: ManifestEntry) -> None:
        self._writer.writerow(asdict(entry))
        self._file.flush()

    def close(self) -> None:
        self._file.close()
