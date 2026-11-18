# AI code reviewer: evaluation

**Question:** On PRs with a known planted bug, how often does the AI reviewer flag it, and how often does it flag problems in clean code?

## Method
- 12 seeded-bug cases plus 4 clean controls in `eval/ai_review/cases/`. Each is a small, realistic file written the way an AI coding assistant might produce it. The labels give the bug's file and line range.
- Bug classes covered: SQL injection, hardcoded secret, off-by-one, missing authorization check, race condition (check-then-act), path traversal, unsafe pickle deserialization, mutable default argument, swallowed exception / resource leak, timing-unsafe compare, date boundary error, SSRF.
- Clean controls are the *fixed* versions of 4 of those cases. That's a hard test for false positives, because the code looks almost the same as the buggy version.
- **A hit** is a finding of medium severity or higher on the labelled file, within ±2 lines of the labelled range.
- **A false positive** is any finding of medium severity or higher that isn't a hit.
- Each case runs N=3 times to show run-to-run variance. Intervals are Wilson 95%, the same as in the RAG retrieval benchmark.
- Model under test: ______ (set via `AI_REVIEW_PROVIDER` / `AI_REVIEW_MODEL`; default is Claude if `ANTHROPIC_API_KEY` is set, otherwise Groq's free `openai/gpt-oss-120b`).
- Command: `python eval/ai_review/run_eval.py --runs 3`

## v1 run (Groq `openai/gpt-oss-120b`, 1 run per case)

| metric | value |
|---|---|
| Recall, textbook bugs | **12/12** (95% CI 76–100%) |
| "False alarms" on clean controls (as v1 scored them) | 2/4 |
| Precision (as v1 scored it) | 79% |

**What v1 actually showed: my scoring was wrong, not the model.** v1 counted *every* finding other than the planted bug as noise. When I read them by hand, most were real:
- On the "clean" locked `reserve()`, the model found that a negative quantity *increases* stock. My control wasn't clean.
- On the planted cases, the extras were real secondary issues: an HMAC with an empty default key, `r.json()` with no status check, unhandled network errors.
- Only one was a genuine false positive: a `KeyError` on a missing secret at import time, which fails fast on purpose.

All 12 hits also named the right bug category, so the line-proximity rule wasn't inflating recall.

**Lessons:**
1. You can't grade an AI reviewer fully automatically. Someone has to judge whether each extra finding is real, so v2 adds `adjudications.json` for human verdicts.
2. 100% recall on textbook cases mostly shows that the test was easy. v2 adds 6 *hard* cases: an IDOR, a cache that leaks data across users, a retry that double-charges, float money, a cursor that skips rows, and naive vs UTC time. None of them contains a "dangerous function" to pattern-match on.

## Results (fill in from results.json)

| metric | value | 95% CI |
|---|---|---|
| Recall (bug runs caught) | __ / 36 | __–__% |
| Clean-PR false-alarm rate | __ / 12 | __–__% |
| Finding precision | __% | |

| bug class | caught (of 3) | notes |
|---|---|---|
| sql_injection | | |
| ... | | |

## Where human judgment is still required
(Write this from the misses. Expected candidates: concurrency bugs that need knowledge of the runtime and threading model, auth checks that depend on app-wide conventions, and business-rule boundaries like the trial-days case, where only the docstring states the intent.)

## Limitations
- n is small. The intervals are wide, and I report them rather than hide them.
- The cases are single-file and new-file only. Real PRs have cross-file context the bot never sees.
- I wrote the cases myself, so they're biased toward bug classes I already know about.
