import re

DIGIT_RUN = re.compile(r"\d+")
MAX_FURNITURE_DIGIT_RUN_LEN = 3


def _skeleton(text: str) -> str:
    stripped = DIGIT_RUN.sub("", text)
    return re.sub(r"\s+", " ", stripped).strip()


def is_page_furniture(text_a: str, text_b: str) -> bool:
    """True if text_b is text_a's page-furniture duplicate: the two texts
    are identical once all digits are stripped and whitespace is collapsed
    (their "skeleton" matches -- this is the primary signal that the two
    are the same repeating template, not coincidentally-similar unrelated
    content), AND the same total count of digit-runs differ only by
    page-number-scale values (each differing run at most
    MAX_FURNITURE_DIGIT_RUN_LEN digits long). A skeleton mismatch alone is
    conclusive: two texts with genuinely different words are never
    furniture, regardless of how their digit-runs happen to compare. A
    matching skeleton with zero digit-runs in either text means the texts
    are byte-identical (nothing was stripped) -- also genuine furniture.
    See docs/superpowers/specs/2026-08-03-eval-harness-design.md section 4
    for the original digit-run analysis (still required as an extra guard
    against theoretical cases like a varying non-page-number field hidden
    inside an otherwise-matching skeleton), and the eval-harness
    implementation plan's Task 8 real-corpus run for the specific
    false-positive/false-negative pairs (unrelated hazard-table entries;
    "Operations" repeated with zero embedded digits) that made the
    skeleton-match precondition necessary.
    """
    if _skeleton(text_a) != _skeleton(text_b):
        return False
    runs_a = DIGIT_RUN.findall(text_a)
    runs_b = DIGIT_RUN.findall(text_b)
    if len(runs_a) != len(runs_b):
        return False
    for run_a, run_b in zip(runs_a, runs_b):
        if run_a != run_b and max(len(run_a), len(run_b)) > MAX_FURNITURE_DIGIT_RUN_LEN:
            return False
    return True
