"""Evaluates the AI reviewer on seeded-bug PRs.

Each case in cases/<name>/ is a file (after.py) presented to the reviewer as
a brand-new file in a PR diff, plus label.json:
  - bug cases:   {"category", "path", "lines": [start, end]}
  - clean cases: {"category": null}  (controls for false positives)

Scoring (deliberately simple and stated up front):
  * HIT   = a finding with severity >= medium on the labelled path whose line
            is within LINE_TOLERANCE of the labelled range.
  * MISS  = bug case with no HIT.
  * Every other medium+ finding is judged by a HUMAN in adjudications.json:
    'valid' extra (a real issue), or 'invalid' (a true false positive).
    Unjudged findings are listed as "needs review" rather than guessed at --
    v1 counted all of them as noise and badly understated precision.
Recall is reported separately for "easy" (textbook) and "hard" (logic /
context-dependent) cases, with Wilson 95% intervals as in the RAG benchmark.

Usage: ANTHROPIC_API_KEY=... python eval/ai_review/run_eval.py [--runs 3]
Writes eval/ai_review/results.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fda_device_rag.eval.metrics import wilson_interval  # noqa: E402
from tools.ai_review.review import SEVERITIES, Finding, review_diff  # noqa: E402

CASES_DIR = Path(__file__).parent / "cases"
ADJ_PATH = Path(__file__).parent / "adjudications.json"
LINE_TOLERANCE = 2
COUNTED = set(SEVERITIES[:3])  # critical, high, medium


def new_file_diff(path: str, source: str) -> str:
    lines = source.splitlines()
    body = "\n".join("+" + line for line in lines)
    return (f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
            f"@@ -0,0 +1,{len(lines)} @@\n{body}\n")


def score_case(label: dict, findings: list[Finding], verdicts: list[dict] | None = None) -> dict:
    """HIT = counted finding on the labelled path within LINE_TOLERANCE of the
    planted bug. Every other counted finding is classified by human verdicts:
    'valid' extra, confirmed false positive, or needs review."""
    counted = [f for f in findings if f.severity in COUNTED]
    hits = []
    if label["category"]:
        lo, hi = label["lines"]
        hits = [f for f in counted if f.path == label["path"]
                and lo - LINE_TOLERANCE <= f.line <= hi + LINE_TOLERANCE]
    valid, invalid, unreviewed = [], [], []
    for f in counted:
        if f in hits:
            continue
        v = next((x["verdict"] for x in (verdicts or []) if x["match"].lower() in f.title.lower()), None)
        (valid if v == "valid" else invalid if v == "invalid" else unreviewed).append(f)
    return {"hit": bool(hits), "category_match": any(f.category == label["category"] for f in hits),
            "valid_extras": len(valid), "false_positives": len(invalid),
            "unreviewed": [f.title for f in unreviewed]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1, help="repeat each case to measure run-to-run variance")
    args = ap.parse_args()

    adjudications = json.loads(ADJ_PATH.read_text()) if ADJ_PATH.exists() else {}
    rows = []
    for case_dir in sorted(p for p in CASES_DIR.iterdir() if p.is_dir()):
        label = json.loads((case_dir / "label.json").read_text())
        diff = new_file_diff(label["path"], (case_dir / "after.py").read_text())
        for run in range(args.runs):
            _, findings = review_diff(diff)
            s = score_case(label, findings, adjudications.get(case_dir.name))
            rows.append({"case": case_dir.name, "run": run, "difficulty": label.get("difficulty"),
                         "category": label["category"], **s, "findings": [asdict(f) for f in findings]})
            print(f"{case_dir.name} run{run}: hit={s['hit']} fp={s['false_positives']} "
                  f"unreviewed={len(s['unreviewed'])}", file=sys.stderr)

    out = Path(__file__).parent / "results.json"
    summary = {}
    for diff_level in ("easy", "hard"):
        rs = [r for r in rows if r["difficulty"] == diff_level]
        h = sum(r["hit"] for r in rs)
        summary[f"recall_{diff_level}"] = {"hits": h, "n": len(rs), "ci95": wilson_interval(h, len(rs))}
    hits = sum(r["hit"] for r in rows if r["category"])
    valid = sum(r["valid_extras"] for r in rows)
    fps = sum(r["false_positives"] for r in rows)
    unrev = sum(len(r["unreviewed"]) for r in rows)
    reviewed = hits + valid + fps
    summary.update({"valid_extras": valid, "false_positives": fps, "unreviewed": unrev,
                    "precision_reviewed": (hits + valid) / reviewed if reviewed else 0.0})
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("| case | difficulty | caught | valid extras | false pos. | needs review |\n|---|---|---|---|---|---|")
    for case in sorted({r["case"] for r in rows}):
        rs = [r for r in rows if r["case"] == case]
        print(f"| {case} | {rs[0]['difficulty']} | {sum(r['hit'] for r in rs)}/{len(rs)} | "
              f"{sum(r['valid_extras'] for r in rs)} | {sum(r['false_positives'] for r in rs)} | "
              f"{sum(len(r['unreviewed']) for r in rs)} |")
    print()
    for lvl in ("easy", "hard"):
        s = summary[f"recall_{lvl}"]
        lo, hi = s["ci95"]
        print(f"Recall ({lvl}): {s['hits']}/{s['n']} (95% CI {lo:.0f}-{hi:.0f}%)")
    print(f"Extra findings: {valid} valid, {fps} false positives, {unrev} need human review")
    print(f"Precision over reviewed findings: {summary['precision_reviewed']:.0%}")
    if unrev:
        print("\nNeeds review (add verdicts to adjudications.json):")
        for r in rows:
            for title in r["unreviewed"]:
                print(f"  {r['case']}: {title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
