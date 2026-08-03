import re

DIGIT_RUN = re.compile(r"\d+")
MAX_FURNITURE_DIGIT_RUN_LEN = 3


def _strip_heading(text: str, heading: str) -> str:
    if heading and text.startswith(heading + "\n"):
        return text[len(heading) + 1:]
    return text


def _skeleton(text: str) -> str:
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

    Both the skeleton and the digit-run comparison operate on each text
    with its own heading already stripped -- the two halves of the rule
    must agree on what "the content" is, otherwise a heading that itself
    contains digits (a real pattern in this corpus, e.g. `Z-800F`) would
    inject a spurious extra/differing digit-run into the comparison after
    the skeleton already matched, silently missing genuine furniture.

    See docs/superpowers/specs/2026-08-03-eval-harness-design.md section 4
    for the round-2 false-positive pairs (unrelated guidance citations;
    needle-set spec tables) that made the skeleton-match precondition
    necessary, and section 4's round 5 for the cross-heading-footer
    evidence (round 1's and round 3's own empirical scripts always grouped
    candidates by heading first, which is what let them find these cases;
    that grouping never made it into this function until round 5, and is
    an implementation-time discovery, not a restoration of a previously
    validated behavior).
    """
    stripped_a = _strip_heading(text_a, heading_a)
    stripped_b = _strip_heading(text_b, heading_b)
    if _skeleton(stripped_a) != _skeleton(stripped_b):
        return False
    runs_a = DIGIT_RUN.findall(stripped_a)
    runs_b = DIGIT_RUN.findall(stripped_b)
    if len(runs_a) != len(runs_b):
        return False
    for run_a, run_b in zip(runs_a, runs_b):
        if run_a != run_b and max(len(run_a), len(run_b)) > MAX_FURNITURE_DIGIT_RUN_LEN:
            return False
    return True
