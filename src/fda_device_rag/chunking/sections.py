import re
from dataclasses import dataclass

NUMBERED_PREFIX = re.compile(r"^\d+(?:(?:\.\d+)+\.?|\.)\s+")
ALLCAPS_BODY = re.compile(r"^[A-Z][A-Z0-9 ,\-/:]{2,80}$")
MASTHEAD_TOKEN = re.compile(r"^[A-Z]{2,6}$")
MIN_ALPHA_TOKEN_LEN = 3
MAX_TITLE_WORDS = 8


@dataclass
class Section:
    heading: str
    body: str


def _has_real_word(body: str) -> bool:
    for token in body.split():
        cleaned = token.rstrip(":,.;")
        if len(cleaned) >= MIN_ALPHA_TOKEN_LEN and cleaned.isalpha():
            return True
    return False


def _is_title_case(body: str) -> bool:
    words = body.rstrip(":").split()
    if not words or len(words) > MAX_TITLE_WORDS:
        return False
    has_long_word = False
    for word in words:
        cleaned = word.strip(",")
        if not cleaned or not cleaned[0].isalpha():
            return False
        if len(cleaned) <= 3:
            continue
        if not cleaned[0].isupper():
            return False
        has_long_word = True
    return has_long_word


def _classify(line: str, seen_real_heading: bool) -> tuple[bool, bool]:
    """Returns (is_heading, updated_seen_real_heading)."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80 or stripped.endswith("."):
        return False, seen_real_heading

    prefix_match = NUMBERED_PREFIX.match(stripped)
    body = stripped[prefix_match.end():] if prefix_match else stripped

    is_allcaps = bool(ALLCAPS_BODY.match(body))
    is_title = _is_title_case(body) if not is_allcaps else False

    if not (is_allcaps or is_title):
        return False, seen_real_heading
    if not _has_real_word(body):
        return False, seen_real_heading

    is_masthead_candidate = (
        not prefix_match
        and is_allcaps
        and " " not in body
        and bool(MASTHEAD_TOKEN.match(body))
    )
    if is_masthead_candidate and not seen_real_heading:
        return False, seen_real_heading

    return True, True


def detect_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    sections: list[Section] = []
    current_heading = "Document"
    current_body_lines: list[str] = []
    seen_real_heading = False

    for line in lines:
        is_heading, seen_real_heading = _classify(line, seen_real_heading)
        if is_heading:
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
