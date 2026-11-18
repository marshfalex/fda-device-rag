# AI code reviewer: evaluation

**Question:** On PRs with a known planted bug, how often does the AI reviewer flag it, and how often does it flag problems in clean code?

## Method
- 12 textbook ("easy") bug cases, 6 "hard" logic/context cases, and 4 clean controls in `eval/ai_review/cases/`. Each is a small, realistic file written the way an AI coding assistant might produce it. The labels give the bug's file and line range.
- Bug classes covered: SQL injection, hardcoded secret, off-by-one, missing authorization check, race condition (check-then-act), path traversal, unsafe pickle deserialization, mutable default argument, swallowed exception / resource leak, timing-unsafe compare, date boundary error, SSRF.
- Clean controls are the *fixed* versions of 4 of those cases. That's a hard test for false positives, because the code looks almost the same as the buggy version.
- **A hit** is a finding of medium severity or higher on the labelled file, within ±2 lines of the labelled range.
- **Every other** finding of medium severity or higher is judged by hand in `adjudications.json`: either a *valid extra* (a real issue) or a *false positive*.
- Each case runs N=3 times to show run-to-run variance. Intervals are Wilson 95%, the same as in the RAG retrieval benchmark.
- Model under test: **Groq `openai/gpt-oss-120b`** (free tier; set via `AI_REVIEW_PROVIDER` / `AI_REVIEW_MODEL`; default is Claude if `ANTHROPIC_API_KEY` is set, otherwise Groq's free `openai/gpt-oss-120b`).
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

## v2 results (Groq `openai/gpt-oss-120b`, 3 runs per case, 66 reviews)

| metric | value | 95% CI |
|---|---|---|
| Recall, textbook bugs | **36/36** | 90–100% |
| Recall, hard bugs | **13/18 (72%)** | 49–88% |
| All planted bugs | 49/54 (91%) | 80–96% |
| Clean-control runs with a real false positive | 3/12 | 9–53% |
| Precision (hits + valid extras) / all judged findings | 58/61 (95%) | 87–98% |

Per hard case, out of 3 runs each:

| case | caught | what the bug is |
|---|---|---|
| IDOR (any user can read any invoice) | 3/3 | missing ownership check |
| `lru_cache` leaks dashboards across users | 3/3 | cache key ignores the current user |
| Retry double-charges a card | 3/3 | non-idempotent POST retried on timeout |
| Naive local time vs UTC expiry | 3/3 | wrong clock |
| **Float money** | **1/3** | `float` dollars with `round()` |
| **Cursor pagination skips rows** | **0/3** | `WHERE created_at < ?` on a non-unique key |

I did 4 adjudications for this run by hand, using the same verdicts as v1 for reworded versions of the same issues. The only genuine false positive is still the import-time `KeyError` on a missing secret. That is intended fail-fast behaviour, and the model raised it in all 3 runs.

## Where human judgment is still required

1. **Bugs that live in the data, not the code.** The model missed the cursor-pagination bug in 3 out of 3 runs, with zero findings each time. The query looks correct. It's only wrong because `created_at` isn't unique, so rows that share a timestamp at a page boundary are skipped silently, with no error. Catching it means knowing something about the data that the diff doesn't show.
2. **Inconsistency.** The model caught the float-money bug in 1 of 3 runs on *identical input*. A reviewer that sometimes catches a bug can't be what you rely on: if a PR passes AI review, that tells you little about issues the model only catches some of the time.
3. **Context outside the diff.** On this project's own PRs, the reviewer raised confident false positives because it only sees the diff. One example: it flagged a division by zero in `wilson_interval`, which already guards `n == 0` in a file the diff didn't include.
4. **Judging the findings themselves.** Most extra findings were real, but some weren't, and the scoring itself was wrong until a human read every finding (see v1). Someone still has to decide which comments to act on.

**Takeaway:** the bot is a cheap, fast first pass that reliably catches textbook vulnerabilities and many logic bugs. Treat it as advisory and keep a human approval on every merge. That's what this repo enforces: the bot can only comment.

## Limitations
- n is small (18 hard runs). The intervals are wide, and I report them rather than hide them.
- One model on its free tier. A larger model (e.g. Claude) may catch more; `AI_REVIEW_PROVIDER=anthropic` reruns the same eval for a comparison.
- The hard-case labels and the adjudications are my own judgement.
- The cases are single-file and new-file only. Real PRs have cross-file context the bot never sees.
- I wrote the cases myself, so they're biased toward bug classes I already know about.
