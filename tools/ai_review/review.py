"""AI code-review gate: sends a PR diff to an LLM (Claude, or Groq's free
tier) and returns structured findings (correctness / security /
maintainability, each with a severity).

Human-in-the-loop by design: this tool can only *comment*. It never
approves, requests changes, or merges -- branch protection requires a
human approval for that. The model is a second pair of eyes, not the gate.

CLI:
  python -m tools.ai_review.review --diff-file pr.diff            # print JSON
  python -m tools.ai_review.review --diff-file pr.diff --post     # post to the PR (CI)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass

# Provider: "anthropic" (Claude) or "groq" (free tier). Defaults to whichever
# API key is present, preferring Anthropic.
PROVIDER = os.environ.get("AI_REVIEW_PROVIDER") or (
    "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "groq")
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "groq": "openai/gpt-oss-120b"}
MODEL = os.environ.get("AI_REVIEW_MODEL", DEFAULT_MODELS.get(PROVIDER, ""))
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_DIFF_CHARS = 400_000  # ~100k tokens; bigger PRs get a "too large" note instead of silent truncation

CATEGORIES = ["correctness", "security", "maintainability"]
SEVERITIES = ["critical", "high", "medium", "low"]

SYSTEM_PROMPT = """You are a senior engineer reviewing a pull request diff.
Report only real, specific problems introduced or exposed by the changed lines:
bugs, security vulnerabilities (injection, secrets, missing authz/authn,
unsafe deserialization, race conditions), and maintainability issues that
would matter to a reviewer. Do not comment on style a linter would catch,
and do not praise. If the diff has no problems, return an empty list.

For each finding give the file path exactly as it appears in the diff, the
line number in the NEW version of the file, a category, a severity, a
one-line title, and a short explanation with a concrete fix.
Severity guide: critical = exploitable or data loss; high = likely bug in
normal use; medium = bug in edge cases or notable risk; low = minor."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "severity": {"type": "string", "enum": SEVERITIES},
                    "title": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["path", "line", "category", "severity", "title", "explanation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "findings"],
    "additionalProperties": False,
}


@dataclass
class Finding:
    path: str
    line: int
    category: str
    severity: str
    title: str
    explanation: str


def parse_review(raw_json: str) -> tuple[str, list[Finding]]:
    """Validates the model's JSON into Findings, dropping malformed items
    rather than failing the whole review."""
    data = json.loads(raw_json)
    findings = []
    for item in data.get("findings", []):
        try:
            category = str(item["category"]).strip().lower()
            severity = str(item["severity"]).strip().lower()
            f = Finding(
                path=str(item["path"]).removeprefix("b/"),
                line=int(item["line"]),
                category=category if category in CATEGORIES else "maintainability",
                severity=severity if severity in SEVERITIES else "low",
                title=str(item["title"]),
                explanation=str(item["explanation"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        findings.append(f)
    return str(data.get("summary", "")), findings


def review_diff(diff: str) -> tuple[str, list[Finding]]:
    """One model call -> (summary, findings), on the configured provider."""
    if len(diff) > MAX_DIFF_CHARS:
        return (f"Diff is {len(diff):,} chars, over the {MAX_DIFF_CHARS:,} review limit; "
                "split the PR or review manually.", [])
    if PROVIDER not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported AI_REVIEW_PROVIDER {PROVIDER!r}; use one of {sorted(DEFAULT_MODELS)}")
    key_var = "GROQ_API_KEY" if PROVIDER == "groq" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_var, "").strip():
        raise RuntimeError(f"{key_var} is not set; it is required for AI_REVIEW_PROVIDER={PROVIDER!r}")
    if PROVIDER == "groq":
        return _review_groq(diff)
    return _review_anthropic(diff)


def _review_anthropic(diff: str) -> tuple[str, list[Finding]]:
    import anthropic  # lazy: keeps the anthropic SDK out of the core test deps

    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=32000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        messages=[{"role": "user", "content": f"<diff>\n{diff}\n</diff>"}],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "refusal":
        return ("Model declined to review this diff; human review required.", [])
    text = next((b.text for b in response.content if b.type == "text"), "")
    return parse_review(text)


def _review_groq(diff: str, retries: int = 4) -> tuple[str, list[Finding]]:
    """Groq's OpenAI-compatible endpoint (free tier). JSON mode guarantees
    valid JSON but not our schema, so the schema goes in the prompt and
    parse_review() drops anything malformed."""
    import time

    import requests

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nRespond with ONLY a JSON object matching this "
             "JSON Schema:\n" + json.dumps(OUTPUT_SCHEMA)},
            {"role": "user", "content": f"<diff>\n{diff}\n</diff>"},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY'].strip()}"}
    for attempt in range(retries):
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=120)
        if r.status_code == 429 and attempt < retries - 1:  # free-tier rate limit
            time.sleep(float(r.headers.get("retry-after", 10 * (attempt + 1))))
            continue
        r.raise_for_status()
        return parse_review(r.json()["choices"][0]["message"]["content"])
    raise RuntimeError("Groq rate limit: retries exhausted")


# ---------------------------------------------------------------- GitHub ---

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def commentable_lines(diff: str) -> dict[str, set[int]]:
    """New-file line numbers that appear in the diff (added or context),
    i.e. the lines GitHub accepts inline review comments on."""
    result: dict[str, set[int]] = {}
    path, new_line = None, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            path = None if target == "/dev/null" else target.removeprefix("b/")
            if path:
                result.setdefault(path, set())
        elif (m := _HUNK_RE.match(line)) and path:
            new_line = int(m.group(1))
        elif path and line.startswith("+") and not line.startswith("+++"):
            result[path].add(new_line)
            new_line += 1
        elif path and line.startswith(" "):
            result[path].add(new_line)
            new_line += 1
    return result


SEVERITY_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}


def format_body(f: Finding) -> str:
    return (f"{SEVERITY_ICON[f.severity]} **{f.severity.upper()} · {f.category}** — {f.title}\n\n"
            f"{f.explanation}\n\n<sub>AI review (advisory only — a human approves merges)</sub>")


def post_review(diff: str, summary: str, findings: list[Finding]) -> None:
    import requests

    repo, pr = os.environ["GITHUB_REPOSITORY"], os.environ["PR_NUMBER"]
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
               "Accept": "application/vnd.github+json"}
    allowed = commentable_lines(diff)
    inline, overflow = [], []
    for f in findings:
        (inline if f.line in allowed.get(f.path, set()) else overflow).append(f)

    counts = {s: sum(f.severity == s for f in findings) for s in SEVERITIES}
    body = ["## 🤖 AI review (advisory)", "", summary or "No summary.", "",
            "| critical | high | medium | low |", "|---|---|---|---|",
            f"| {counts['critical']} | {counts['high']} | {counts['medium']} | {counts['low']} |", ""]
    for f in overflow:
        body.append(f"- `{f.path}:{f.line}` {format_body(f)}")
    body.append("\n_The AI can only comment. Merging requires a human approval._")

    payload = {
        "event": "COMMENT",  # never APPROVE / REQUEST_CHANGES
        "body": "\n".join(body),
        "comments": [{"path": f.path, "line": f.line, "side": "RIGHT", "body": format_body(f)} for f in inline],
    }
    url = f"https://api.github.com/repos/{repo}/pulls/{pr}/reviews"
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code == 422 and inline:  # a line GitHub rejected: fall back to summary-only
        payload["body"] += "\n\n" + "\n".join(f"- `{f.path}:{f.line}` {format_body(f)}" for f in inline)
        payload["comments"] = []
        r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-file", required=True)
    ap.add_argument("--post", action="store_true", help="post as a PR review (needs GITHUB_* env)")
    ap.add_argument("--fail-on", choices=SEVERITIES, help="exit 1 if any finding at/above this severity")
    args = ap.parse_args()
    with open(args.diff_file, encoding="utf-8") as fh:
        diff = fh.read()
    summary, findings = review_diff(diff)
    print(json.dumps({"summary": summary, "findings": [asdict(f) for f in findings]}, indent=2))
    if args.post:
        post_review(diff, summary, findings)
    if args.fail_on:
        cutoff = SEVERITIES.index(args.fail_on)
        if any(SEVERITIES.index(f.severity) <= cutoff for f in findings):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
