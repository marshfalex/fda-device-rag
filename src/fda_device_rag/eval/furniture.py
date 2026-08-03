import re

DIGIT_RUN = re.compile(r"\d+")
MAX_FURNITURE_DIGIT_RUN_LEN = 3


def is_page_furniture(text_a: str, text_b: str) -> bool:
    """True if text_b is text_a's page-furniture duplicate: the same total
    count of digit-runs, and every digit-run that differs between the two is
    at most MAX_FURNITURE_DIGIT_RUN_LEN digits long (page-number-scale). A
    4+-digit differing run, or a differing digit-run count, means the two
    are distinct real content and must never be collapsed -- see
    docs/superpowers/specs/2026-08-03-eval-harness-design.md section 4 for
    the two false-positive classes (citation numbers, needle-set spec
    tables) this boundary was empirically set to exclude, and the
    monotonicity/isolation checks that confirmed the boundary itself.
    """
    runs_a = DIGIT_RUN.findall(text_a)
    runs_b = DIGIT_RUN.findall(text_b)
    # Furniture requires page-number indicators; texts with no digits can't be furniture
    if not runs_a:
        return False
    if len(runs_a) != len(runs_b):
        return False
    for run_a, run_b in zip(runs_a, runs_b):
        if run_a != run_b and max(len(run_a), len(run_b)) > MAX_FURNITURE_DIGIT_RUN_LEN:
            return False
    return True
