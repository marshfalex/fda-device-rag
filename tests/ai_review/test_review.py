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


def test_parse_review_normalizes_case():
    raw = json.dumps({"summary": "", "findings": [
        {"path": "a.py", "line": 3, "category": "Security", "severity": "HIGH", "title": "t", "explanation": "e"}]})
    _, findings = parse_review(raw)
    assert (findings[0].category, findings[0].severity) == ("security", "high")


def test_groq_provider_posts_json_mode_and_retries_429(monkeypatch):
    import requests

    from tools.ai_review import review

    calls = []

    class Resp:
        def __init__(self, status, body=None):
            self.status_code, self._body, self.headers = status, body, {"retry-after": "0"}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return self._body

    content = json.dumps({"summary": "s", "findings": [
        {"path": "a.py", "line": 1, "category": "security", "severity": "high", "title": "t", "explanation": "e"}]})
    responses = [Resp(429), Resp(200, {"choices": [{"message": {"content": content}}]})]

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return responses.pop(0)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(review, "PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    summary, findings = review.review_diff("+x = 1\n")
    assert summary == "s" and findings[0].severity == "high"
    assert len(calls) == 2 and calls[0]["response_format"] == {"type": "json_object"}


def test_review_diff_rejects_unknown_provider_and_missing_key(monkeypatch):
    import pytest

    from tools.ai_review import review

    monkeypatch.setattr(review, "PROVIDER", "openai")
    with pytest.raises(ValueError, match="Unsupported AI_REVIEW_PROVIDER"):
        review.review_diff("+x\n")
    monkeypatch.setattr(review, "PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        review.review_diff("+x\n")
