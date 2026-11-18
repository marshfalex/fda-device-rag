import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "ai_review"))

from run_eval import new_file_diff, score_case  # noqa: E402

from tools.ai_review.review import (  # noqa: E402
    Finding,
    commentable_lines,
    parse_review,
)


def _f(path="app/x.py", line=5, severity="high", category="security"):
    return Finding(path, line, category, severity, "t", "e")


def test_parse_review_drops_malformed_and_normalizes():
    raw = json.dumps({"summary": "s", "findings": [
        {"path": "b/app/x.py", "line": "7", "category": "security", "severity": "high", "title": "t", "explanation": "e"},
        {"path": "app/x.py"},  # malformed -> dropped
        {"path": "a.py", "line": 1, "category": "weird", "severity": "nope", "title": "t", "explanation": "e"},
    ]})
    summary, findings = parse_review(raw)
    assert summary == "s"
    assert [(f.path, f.line) for f in findings] == [("app/x.py", 7), ("a.py", 1)]
    assert findings[1].category == "maintainability" and findings[1].severity == "low"


def test_commentable_lines_tracks_new_file_numbers():
    diff = ("diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -10,3 +10,4 @@\n"
            " ctx\n-old\n+new1\n+new2\n ctx2\n")
    assert commentable_lines(diff) == {"f.py": {10, 11, 12, 13}}


def test_new_file_diff_roundtrips_line_numbers():
    diff = new_file_diff("app/x.py", "a\nb\nc\n")
    assert commentable_lines(diff) == {"app/x.py": {1, 2, 3}}


def test_score_hit_within_tolerance_and_low_severity_ignored():
    label = {"category": "security", "path": "app/x.py", "lines": [5, 5]}
    assert score_case(label, [_f(line=7)])["hit"] is True
    assert score_case(label, [_f(line=9)])["hit"] is False
    s = score_case(label, [_f(line=5, severity="low")])
    assert s["hit"] is False and s["false_positives"] == 0


def test_score_clean_case_counts_false_positives():
    label = {"category": None, "path": "app/x.py", "lines": None}
    assert score_case(label, [_f(), _f(severity="medium")])["false_positives"] == 2
