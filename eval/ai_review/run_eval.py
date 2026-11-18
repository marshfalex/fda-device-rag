"""Evaluates the AI reviewer on seeded-bug PRs.

Each case in cases/<name>/ is a file (after.py) presented to the reviewer as
a brand-new file in a PR diff, plus label.json:
  - bug cases:   {"category", "path", "lines": [start, end]}
  - clean cases: {"category": null}  (controls for false positives)

Scoring (deliberately simple and stated up front):
  * HIT   = a finding with severity >= medium on the labelled path whose line
            is within LINE_TOLERANCE of the labelled range.
  * MISS  = bug case with no HIT.
  * FALSE POSITIVE = any finding with severity >= medium that is not a HIT
            (on clean cases, every such finding).
Recall and clean-case false-alarm rate get Wilson 95% intervals, same as the
RAG retrieval benchmark (fda_device_rag.eval.metrics.wilson_interval).

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
LINE_TOLERANCE = 2
COUNTED = set(SEVERITIES[:3])  # critical, high, medium


def new_file_diff(path: str, source: str) -> str:
    lines = source.splitlines()
    body = "\n".join("+" + line for line in lines)
    return (f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
            f"@@ -0,0 +1,{len(lines)} @@\n{body}\n")


def score_case(label: dict, findings: list[Finding]) -> dict:
    counted = [f for f in findings if f.severity in COUNTED]
    hits = []
    if label["category"]:
        lo, hi = label["lines"]
        hits = [f for f in counted if f.path == label["path"]
                and lo - LINE_TOLERANCE <= f.line <= hi + LINE_TOLERANCE]
    fps = [f for f in counted if f not in hits]
    return {"hit": bool(hits), "false_positives": len(fps),
            "category_match": any(f.category == label["category"] for f in hits)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1, help="repeat each case to measure run-to-run variance")
    args = ap.parse_args()

    rows = []
    for case_dir in sorted(p for p in CASES_DIR.iterdir() if p.is_dir()):
        label = json.loads((case_dir / "label.json").read_text())
        path = label["path"]
        diff = new_file_diff(path, (case_dir / "after.py").read_text())
        for run in range(args.runs):
            _, findings = review_diff(diff)
            s = score_case(label, findings)
            rows.append({"case": case_dir.name, "run": run, "bug_class": label["bug_class"],
                         "category": label["category"], **s, "findings": [asdict(f) for f in findings]})
            print(f"{case_dir.name} run{run}: hit={s['hit']} fp={s['false_positives']}", file=sys.stderr)

    bug = [r for r in rows if r["category"]]
    clean = [r for r in rows if not r["category"]]
    hits = sum(r["hit"] for r in bug)
    clean_alarms = sum(r["false_positives"] > 0 for r in clean)
    total_fp = sum(r["false_positives"] for r in rows)
    total_findings = sum(len([f for f in r["findings"] if f["severity"] in COUNTED]) for r in rows)

    summary = {
        "recall": hits / len(bug) if bug else 0.0, "recall_ci95": wilson_interval(hits, len(bug)),
        "clean_false_alarm_rate": clean_alarms / len(clean) if clean else 0.0,
        "clean_false_alarm_ci95": wilson_interval(clean_alarms, len(clean)),
        "precision": (total_findings - total_fp) / total_findings if total_findings else 0.0,
        "n_bug_runs": len(bug), "n_clean_runs": len(clean),
    }
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print("| case | caught | false positives |\n|---|---|---|")
    for case in sorted({r["case"] for r in rows}):
        rs = [r for r in rows if r["case"] == case]
        print(f"| {case} | {sum(r['hit'] for r in rs)}/{len(rs)} | {sum(r['false_positives'] for r in rs)} |")
    lo, hi = summary["recall_ci95"]
    flo, fhi = summary["clean_false_alarm_ci95"]
    print(f"\nRecall: {summary['recall']:.0%} (95% CI {lo:.0f}–{hi:.0f}%), n={len(bug)}")
    print(f"Clean-PR false alarm rate: {summary['clean_false_alarm_rate']:.0%} (95% CI {flo:.0f}–{fhi:.0f}%), n={len(clean)}")
    print(f"Finding precision: {summary['precision']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
