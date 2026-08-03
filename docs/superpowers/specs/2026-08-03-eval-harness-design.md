# Evaluation Harness: Frozen 50-Question Benchmark — Design

Status: Approved for planning
Date: 2026-08-03

## 1. Problem

Phase 2 (real corpus population + section-detection fixes) is merged. The architecture
doc's §6 commits to a leakage-safe, taxonomy-grounded 50-question benchmark ("questions
and their gold source-document IDs are authored from FDA's actual complaint/problem
taxonomy fields before the retriever ever runs against them, and the question file is
frozen prior to any retrieval scoring") but leaves the authoring mechanics — sampling,
gold-ID definition, phrasing, file format, and workflow — undesigned. This document
specifies all of them.

**Explicit scope decision carried through this whole design:** questions whose gold
source lands in one of `188844`/`153781`'s mislabeled `Contains Nonbinding
Recommendations` chunks (§4 of the section-detection-fix design; up to 41% of a
document's chunks share this uninformative label even though the underlying chunk text
is real and correct) are not excluded from sampling. Excluding them would make the
benchmark corpus artificially cleaner than the real corpus actually is. Every mechanism
below (sampling, gold-ID keying, the `notes` field) is designed to make this inclusion
correct and visible, not to work around it.

## 2. Question-Set Composition

50 questions total, stratified by source type to roughly match the architecture doc's
~12-13-per-type target:

| Source type | Count | Structure |
|---|---|---|
| Recall | 13 | 1 per top-13-frequency `root_cause_description` category |
| MAUDE | 13 | 1 per top-13-frequency `product_problems` category |
| Guidance PDF | 12 | 3 per document × 4 documents (`78369`, `188844`, `153781`, `73141`) |
| IFU PDF | 12 | 4 per document × 3 documents (`Z-800F`, `intera-3000`, `FreedomEdge`) |

**Category selection (recall/MAUDE): top-frequency, not stratified to include rare
categories.** Rare/edge categories are deliberately deferred to future work rather than
diluting an n=50 benchmark across two different distributions with two different
stories to explain in an interview.

**Document selection (PDF): even split per document, not frequency-weighted.** An
uneven draw risks under-representing or excluding `188844`/`153781` — the two documents
carrying the known mislabeling limitation — which would silently contradict the
inclusion decision in §1.

## 3. Candidate Selection Mechanism

Programmatic, seeded (`random.Random(seed)`), without-replacement, per-category or
per-document — never manual picking. Manually choosing "the representative record"
risks unconsciously favoring cleaner, more retrieval-friendly narrative text, which is
the same category of leakage risk as writing questions after seeing retrieval results,
just one step earlier in the pipeline.

### 3a. Recall / MAUDE record selection

Per category, pipeline order is: **filter candidates to those meeting a pre-declared
minimum narrative-length floor → sort by stable record ID → `rng.choice()` from the
filtered set.** Filtering before the random draw keeps selection reproducible and
non-cherry-picked; drawing first and rejecting-and-rerolling after inspection would
reintroduce manual judgment into a step designed to be free of it.

**Floor validated empirically, not guessed:** checked all 26 categories (13 recall +
13 MAUDE) in the real pulled corpus (`data/raw/recalls.json`, `data/raw/events.json`,
500 records each) at FLOOR=100 characters — zero records excluded in any category.
Global minimums across the full pull: 435 characters (recall), 124 characters (MAUDE);
the 5 shortest records in each category were individually inspected and are genuinely
usable narrative text. The floor is therefore currently non-load-bearing on this
specific data pull, but is formalized as an explicit, reproducible pipeline step (not
a one-time manual check) so it holds if the corpus is ever re-pulled with different
records.

### 3b. PDF section selection

Per document, the sampling pool is every eligible section **instance** — keyed by
`(document_title, first_chunk_index)`, never by `section_name` alone. `section_name`
is a confirmed non-unique label for exactly the population this design must sample
correctly from: `188844` has 13 separate instances sharing the label `Contains
Nonbinding Recommendations`, each a distinct span of real content. Keying or sorting
by `section_name` would collapse these into one candidate instead of 13, reintroducing
the exact swallowing failure mode already fixed at the chunking layer. Verified: all
13 instances confirmed as separate, individually-selectable pool entries.

**Eligibility floor: 72 characters**, not the chunk-level 40-character filter (a
different threshold for a different purpose — that one guards individual chunks
against BM25's short-document bias; this one guards whole section instances against
being trivial/near-empty candidates for question-authoring). Validated against the
full range, not bookend samples: inspected every section instance from 46 characters
(smallest) up through 260 characters across all 7 documents (618 total instances, 20
of them under 72 characters). The under-72 band is unambiguous page-footer/running-
header noise — confirmed by inspecting all 20 instances directly, not a bookend
sample. The 72-149 character band (125 instances) is genuinely mixed — critically, it
contains a real safety warning (`CAUTION: → "Do not continuously use the Zyno
Administration set in the pump more than 72 hours."`, 92 characters) that a
higher/bookend-only-validated floor (150, initially proposed) would have wrongly
excluded. 72 characters is the exact boundary of the confirmed-unambiguous-noise band.

**Pure-furniture skip mechanism:** if the specific instance drawn from the eligible
pool is pure page-furniture (defined mechanically below), skip it and advance to the
next draw in the same deterministic RNG sequence — never draw fresh, and log
`(document, skipped_index, matched_against_index)` for every skip. This is a
sequential redraw, not a second length threshold, because length alone cannot
separate "repeated footer" from "short real content" (the CAUTION warning above proves
short ≠ noise).

## 4. The Page-Furniture Rule

Required to be a precise, pre-declared mechanical rule, not a runtime judgment call —
a judgment call at the moment of application would reintroduce the same
subjective-selection risk this mechanism exists to avoid.

**Rule (final, after two rounds of empirical correction — see below):**

> An eligible instance (≥72 chars) is page-furniture only if another instance in the
> same document has (a) the same total count of digit-runs (`\d+` matches), and (b)
> every digit-run that differs between the two is ≤3 digits long. If digit-run counts
> differ, or any differing run is 4+ digits, the instances are kept distinct and never
> collapsed.

Rationale: furniture (running headers/footers) repeats near-verbatim except for an
embedded page number; real content never does. A page number is well-modeled as a
short (≤3 digit), isolated, monotonically-increasing-with-document-order value — not
merely "a short digit run," which is the distinction the validation below exists to
confirm.

**Round 1 (naive rule, rejected): strip all digits, compare normalized text.** Applied
to all 598 eligible instances across 7 documents: 62 flagged as furniture, 536 kept.
Manual inspection of sample furniture groups (`'Operations' × 7`, `'GETTING STARTED' ×
8/×12`, `'Page ii' × 2`, etc.) confirmed genuine, and an inverse check against 4
known-real examples (`CAUTION:`, `Flow Rates`, `LOADING IV SET INTO Z-800F PUMP`,
`SAVE INFUSION PARAMETERS AS PROTOCOL`) confirmed none were wrongly flagged.

**Round 2 (gap found): blanket digit-stripping collides distinct real facts that
differ only by digits.** Re-inspecting the 13 flagged groups' original (pre-strip)
text found 3 groups (11 of the 62 flagged instances) were **false positives**:

- `78369`: two different real guidance-document citations (`.../ucm073778.htm` "Off-
  the-Shelf Software" vs. `.../ucm259748.htm` "Medical Device Design") collapsed
  because their only difference is a 6-digit number.
- `FreedomEdge_Domestic_IFU_347201_Rev_B`: two groups of genuinely different
  needle-set spec tables (3 instances: Single/Two/Three-Needle; 6 instances:
  Single/Two/Three/Four/Five/Six-Needle Sets), each with distinct item numbers and
  residual volumes (e.g. `12406` vs. `22406`), collapsed because digit-stripping
  erases exactly the values that make the tables distinct.

This is a structural property of digit-stripping, not a coincidence: any corpus
content that differs *only* in numeric values (spec tables, dosages, dimensions) is at
risk, and this corpus has exactly that content. Fixed by replacing blanket
digit-stripping with the position-wise, run-length-bounded comparison in the rule
above (≤3-digit differing runs = furniture; 4+ digit differing runs = distinct real
content).

**Round 3 (full-pool re-validation of the ≤3-digit boundary itself):** the refined
rule's ≤3-digit threshold was itself only an assumption that short digit runs behave
like page numbers — not yet verified as actually functioning as page numbers in
context, and this corpus's real spec-table content (needle gauges, flow rates) is
exactly the content where two genuinely different real facts could differ by only 1-3
digits, the same collision category found twice above, just not yet checked at the
short end. Re-ran the refined rule against the full 598-instance eligible pool (162
pairwise collapses flagged, covering 51 instances in 10 clusters — the same 51
genuine-furniture instances as round 1, with the 11 round-2 false positives now
correctly excluded) and added two independent checks against every one of the 162
flagged pairs:

- **Monotonicity:** does the differing digit-run value change in the same direction as
  chunk/document order (the way a real page number must)? **0/162 violations.**
- **Isolation:** is the differing run isolated (surrounded by whitespace/punctuation,
  not embedded inside a longer alphanumeric code like `RMS12406` or `ucm073778`, the
  way a real page number sits in a footer but an item code does not)? **0/162
  violations** — every differing run sits in a consistent footer-line position (e.g.
  immediately after the section heading, or immediately before the document
  title/part-number line).

Zero violations across both independent checks, on the full pool, after two prior
rounds each found and closed a real gap, is the signal to lock the rule rather than
continue iterating — this round found nothing new.

## 5. Gold-ID Schema

**Structured sources (recall, MAUDE):**
```json
{"record_id": "<the FDA record's own unique ID>"}
```
`record_id` is already carried on every structured chunk's metadata and is untouched
by chunking changes.

**PDF sources (guidance, IFU):**
```json
{"document_title": "188844", "section_name": "Contains Nonbinding Recommendations", "instance_ordinal": 5}
```
`instance_ordinal` is the Nth occurrence (1-indexed, document order) of that heading
text — the same instance-keying used for sampling (§3b), expressed as a
human-readable ordinal rather than a raw chunk index, which would be a meaningless
implementation detail to anyone reading the frozen file.

**Resolution happens at eval-run time, not authoring time:** re-run section detection
and chunking against whatever the corpus currently is, re-aggregate into section
instances, locate the Nth occurrence of the heading in the document, and take
`gold_chunk_ids` = every chunk ID currently belonging to that instance. This is why the
frozen file stores semantic identity instead of chunk IDs directly — chunk boundaries
have already shifted once in this project (the `EFFECTIVE_CHUNK_SIZE` change), and
pinning literal chunk-ID lists would silently invalidate part of a frozen, committed
benchmark on the next such change with no signal that it happened. A locator resolved
at eval time means only a change to section *detection* itself (a much rarer event,
already covered by its own test suite) could affect resolution.

**Resolution failure is a hard, visible error, not a silent skip or a silent miss:** if
a locator can't be resolved (heading renamed, document dropped, ordinal now
out-of-range), it's logged as an explicit resolution error for that question. Eval
output reports the adjusted denominator plainly wherever this happens — e.g. `"47/48
valid questions (2 resolution errors), Hit Rate@5: 39/47 (83.0%)"` — never a
percentage silently computed over a shrunk n.

**Scoring (per architecture doc §6, restated with the instance-keyed gold set):** a
question is a hit if any `gold_chunk_id` appears in the fused top-5. `gold_set_size =
len(gold_chunk_ids)` is logged per question; questions with an outlier gold-set size
are flagged explicitly in the eval README rather than averaged away — a 1-chunk gold
set and a 6-chunk gold set are not equally hard to hit by construction.

## 6. Question Phrasing Style

Natural user-style phrasing (matching `query.py`'s example usage), not
clinical/keyword phrasing. This is the methodologically load-bearing choice, not a
cosmetic one: keyword-style phrasing is structurally closer to what BM25 already
rewards, which would bias the retrieval-ablation comparison (dense-only vs. BM25-only
vs. hybrid) toward BM25 specifically — a confound in the one comparison this eval
exists to settle honestly.

Constraints while drafting: each question must be single and unambiguous even in
natural phrasing (one question → one gold record/section → a clean binary Hit Rate@5
signal), and any specific device/product name must be pulled directly from the
candidate's actual fields, never invented plausible-sounding detail not present in the
source. Tabular spec content (needle-size tables, etc.) is phrased to match its actual
shape rather than triggering a redraw — treating real table data as legitimately
answerable is consistent with not laundering the benchmark away from what the corpus
actually contains.

## 7. Frozen File Format

**Location:** `data/eval/questions.json`. Not gitignored — only `data/raw/`,
`data/chroma/`, `data/manifest.csv`, and `data/bm25_index.pkl` are excluded (see
`.gitignore`), consistent with `data/` already holding both raw (excluded) and derived,
committed artifacts.

**Top level:**
```json
{
  "version": 1,
  "frozen_date": "2026-08-03",
  "questions": [ /* 50 objects */ ]
}
```

**Per question:**
```json
{
  "question_id": "recall-01",
  "source_type": "recall",
  "category": "Device Design/Description",
  "question": "Why was the XYZ infusion pump recalled?",
  "gold": {"record_id": "REC-2026-001234"},
  "notes": null
}
```
- `question_id`: `{source_type}-{seq}`; unique and human-readable, carries no
  chunking-implementation information.
- `category`: the sampled `root_cause_description`/`product_problems` value for
  recall/MAUDE; `null` for guidance/IFU (sampled by document, not taxonomy — §2).
- `gold`: structured-record or PDF-section-instance locator, per §5.
- `notes`: free text; used specifically to flag when a question's gold section is one
  of the known-mislabeled `Contains Nonbinding Recommendations` instances, per §1 —
  making the inclusion decision a literal, greppable field rather than an implicit
  fact about the corpus.

**Freeze mechanics — mechanically enforced, not just documented:**
1. Authored progressively during the §8 collaborative session.
2. Once all 50 are drafted and reviewed, committed to git in one commit.
3. The eval-runner script refuses to run if `data/eval/questions.json` has
   uncommitted changes (`git diff --quiet` check at startup) — leakage can't happen by
   accident even if the procedural rule is forgotten.
4. Every eval run logs the git blob hash of the frozen file
   (`git rev-parse HEAD:data/eval/questions.json`) alongside its results. The
   diff-quiet check alone proves the working tree is clean *at run time*; it cannot
   distinguish a genuine one-time freeze from "saw bad results, edited, recommitted,
   reran." A hash stamped on every run gives a permanent, defensible trail instead of
   only a procedural promise.

## 8. Collaborative Authoring Workflow

**Step 1 — Deterministic sampling script** (built during execution, not this design):
implements §3's pipeline (floor filter → stable-ID sort → seeded `rng.choice()` for
recall/MAUDE; per-document without-replacement draw with sequential redraw-on-furniture
per §3b/§4 for PDF). Outputs an intermediate, uncommitted **candidates file** — 50
slots, each carrying the selected record's raw fields or the selected section
instance's full text plus its resolved locator. This file is not the benchmark; it
exists purely so the phrasing step (Step 3) never needs to browse the live corpus,
which would reintroduce cherry-picking risk one step later in the pipeline.

**Step 2 — Section-diversity check:** verify no two candidates within the same
document's quota share a `section_name` label. Should already hold structurally (the
furniture rule separates same-labeled-but-distinct instances, and without-replacement
sampling can't draw the same instance twice) — this step is a hard-error guard on that
invariant, not a mechanism expected to fire.

**Step 3 — Collaborative phrasing session**, category by category in fixed order:
recall (by category, frequency order) → MAUDE → guidance (by document) → IFU (by
document). For each candidate, a question is drafted grounded only in that candidate's
actual fields/text (§6), reviewed and approved or edited, then appended to the growing
`data/eval/questions.json` draft. Any candidate whose gold section is a known-mislabeled
heading gets its `notes` field populated at this step.

**Step 4 — Freeze**, per §7.

**Design/plan boundary:** this document specifies the pipeline logic and file formats;
the sampling-script implementation and the interactive per-question session are
execution-phase work, following the same design → plan → execution split as every
prior phase in this project.

## 9. Scope

**In scope:** a new sampling script (`scripts/sample_eval_candidates.py`, matching the
existing `scripts/pull_corpus.py`/`build_index.py`/`query.py` convention), the
frozen-file schema and location (`data/eval/questions.json`), the gold-locator
resolution logic (`src/fda_device_rag/eval/gold_resolution.py`, matching the existing
`src/fda_device_rag/{module}/` package convention), and the freeze-enforcement check
in the eventual `scripts/run_eval.py`. Exact module boundaries remain subject to
refinement in the implementation plan; the logic and contracts they must implement do
not.

**Out of scope, explicitly:** the eval-runner script's retrieval-scoring logic itself
(Hit Rate@5/MRR computation, ablation, stratified reporting) — that consumes this
design's frozen file and gold-resolution contract but is its own implementation unit,
per architecture doc §6. The generation-faithfulness spot-check (also architecture doc
§6) is a separate downstream step operating on a subset of this frozen file, not
designed here. Rare/edge taxonomy categories (§2) and any second benchmark iteration
are deferred to future work.

## 10. Testing

New tests required (implementation phase):

- Sampling pipeline: floor filter excludes sub-threshold candidates before the random
  draw; stable-ID sort produces a deterministic pool order; same seed produces the
  same draw across runs.
- PDF instance pool: confirms non-unique `section_name` instances remain separate,
  individually-selectable pool entries (regression guard for the collapsing failure
  mode found in §3b).
- Furniture rule: same-digit-count/≤3-digit-differing-run pairs flagged as furniture;
  differing-digit-count or 4+-digit-differing-run pairs kept distinct (regression
  guard for the two false-positive classes found in §4, round 2); sequential redraw on
  a flagged furniture instance advances through the same RNG sequence rather than
  drawing fresh.
- Gold-locator resolution: structured `record_id` resolves to all chunks sharing that
  ID; PDF locator resolves to the Nth section instance's current chunk IDs; resolution
  failure (bad ordinal, missing document) raises/logs an explicit error rather than
  returning an empty or partial result silently.
- Freeze enforcement: eval-runner refuses to run against an uncommitted
  `questions.json`; logs the correct git blob hash for a committed file.

No changes to existing `Chunk`/`ChunkMetadata`/`Section` interfaces or to the
retrieval pipeline (Tasks 9-13) — this is new, additive functionality (a sampling
script, a gold-resolution module, a frozen data file) alongside the existing
ingestion/retrieval code, not a modification to it.
