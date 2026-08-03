import re

DIGIT_RUN = re.compile(r"\d+")
MAX_FURNITURE_DIGIT_RUN_LEN = 3


def _skeleton(text: str, heading: str) -> str:
    if heading and text.startswith(heading + "\n"):
        text = text[len(heading) + 1:]
    stripped = DIGIT_RUN.sub("", text)
    return re.sub(r"\s+", " ", stripped).strip()


def is_page_furniture(text_a: str, heading_a: str, text_b: str, heading_b: str) -> bool:
    """True if text_b is text_a's page-furniture duplicate.

    Each instance's own heading is stripped from its own text before
    comparison (a repeating page footer commonly attaches to a different
    real section heading each time it recurs -- e.g. the same footer
    trailing "GETTING STARTED" on one page and "INFUSION MODE INFORMATION"
    on another -- so comparing full text including heading would miss this
    genuine furniture pattern, and comparing only the heading-stripped body
    is what correctly recognizes it). After heading-stripping, the two
    texts' "skeleton" (digits stripped, whitespace collapsed) must be
    identical -- this is the primary signal that the two share the same
    repeating template, not coincidentally-similar unrelated content, and
    alone is sufficient to reject any pair with genuinely different words
    (whether the difference is in the heading, the body, or both). Given a
    matching skeleton, the digit-runs additionally must have the same count
    with every differing run at most MAX_FURNITURE_DIGIT_RUN_LEN digits long
    (page-number-scale) -- a residual guard against a matching skeleton
    whose only difference is some other, non-page-number embedded value
    (e.g. a long serial or part number).

    See docs/superpowers/specs/2026-08-03-eval-harness-design.md section 4
    for the original digit-run analysis and the round-2 false-positive
    pairs (unrelated guidance citations; needle-set spec tables) that made
    the skeleton-match precondition necessary, and section 4's round-1/
    round-3 empirical validation for the heading-stripped comparison this
    function restores.
    """
    if _skeleton(text_a, heading_a) != _skeleton(text_b, heading_b):
        return False
    runs_a = DIGIT_RUN.findall(text_a)
    runs_b = DIGIT_RUN.findall(text_b)
    if len(runs_a) != len(runs_b):
        return False
    for run_a, run_b in zip(runs_a, runs_b):
        if run_a != run_b and max(len(run_a), len(run_b)) > MAX_FURNITURE_DIGIT_RUN_LEN:
            return False
    return True
