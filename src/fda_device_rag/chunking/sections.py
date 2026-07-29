import re
from dataclasses import dataclass

HEADING_PATTERN = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][A-Z0-9 ,\-/:]{2,80})$")


@dataclass
class Section:
    heading: str
    body: str


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and len(stripped) <= 80 and bool(HEADING_PATTERN.match(stripped))


def detect_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    sections: list[Section] = []
    current_heading = "Document"
    current_body_lines: list[str] = []

    for line in lines:
        if _is_heading(line):
            if current_body_lines:
                sections.append(Section(current_heading, "\n".join(current_body_lines).strip()))
            current_heading = line.strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)

    if current_body_lines:
        sections.append(Section(current_heading, "\n".join(current_body_lines).strip()))

    non_empty = [s for s in sections if s.body]

    # A running header/footer repeated on every page of a PDF looks like a new
    # heading on each page, fragmenting one logical section into many
    # identically-named tiny ones. Merge consecutive sections that share the
    # exact same heading text back into a single section.
    merged: list[Section] = []
    for section in non_empty:
        if merged and merged[-1].heading == section.heading:
            merged[-1] = Section(section.heading, f"{merged[-1].body}\n{section.body}")
        else:
            merged.append(section)

    return merged
