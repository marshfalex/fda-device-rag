# Section Detection: Title-Case Heading Fix + Chunk-Length Filter — Design

Status: Approved for planning
Date: 2026-07-29

## 1. Problem

Populating the real guidance/IFU PDF corpus for the first time (Phase 2 prep) exposed
that `detect_sections` (src/fda_device_rag/chunking/sections.py) only recognizes
ALL-CAPS heading lines. Running the existing chunker against all 7 real documents and
measuring the fraction of a document's chunks landing under one section label found:

| Document | Chunks under one label | Root cause |
|---|---|---|
| `78369` (Infusion Pump TPLC guidance) | 100% (69/69) | Zero headings detected — this document's real headings (`Preface`, `Table of Contents`, `Public Comment`) are Title Case, not ALL CAPS. |
| `188844` (Computer Software Assurance guidance) | 97% (73/75) | A handful of ALL-CAPS masthead acronym fragments (`CDRH`, `CBER`, `SOP`) fire as headings near the top; since no further heading is ever detected (same Title-Case blind spot), the last one becomes a catch-all sink for the rest of the document. |
| `153781` (Premarket Submission Content guidance) | 94% (83/88) | Same mechanism as `188844` (`CDER`, `AND`). |

Tracing the detected-section order confirmed this is **one root cause manifesting two
ways** (total blindness vs. blindness-after-a-few-incidental-acronym-matches), not two
separate bugs. `Z-800F` and `intera-3000` (IFUs) were previously verified correct and
must not regress.

## 2. Fix, Part 1: Heading Classifier (`_is_heading` replacement)

Three-part classifier, in priority order:

**(a) Title-Case detection — load-bearing.** After stripping any numbered prefix
(unchanged), a line qualifies if it has ≤8 words, no word longer than 3 letters is
lowercase (short connectors like "of"/"for"/"and" may stay lowercase — `Table of
Contents` passes), and at least one word is long enough to matter. This is what fixes
all three documents above.

**(b) Real-word content guard — required for both (a) and any ALL-CAPS match.** At
least one whitespace-separated token, after stripping trailing punctuation, must be
3+ *purely alphabetic* characters. This rejects catalog-code/table rows like `F120
F180 F275...` (every token mixes letters and digits) — fixes the FreedomEdge
false-positive as a side effect of the same guard, no separate code path.

**(c) Masthead suppression — secondary polish, position-scoped, not length-scoped.**
A candidate is only suppressed if it is a bare single ALL-CAPS token, 2-6 characters,
with no numbered prefix, **and no real (non-suppressed) heading has been accepted
anywhere earlier in the document yet.** Once the first real heading fires, this rule
never applies again for the rest of the document. Hand-traced against all 4 previously
non-broken documents (Z-800F, intera-3000, FreedomEdge, 73141) with no case found where
this would suppress a legitimate heading.

**Two additional guards found necessary during empirical validation (§3):**

**(d) Reject any candidate whose text ends in a literal period.** Fixes two failure
modes surfaced by running the classifier against real text: wrapped sentence/title
fragments (`"Pump."`, `"Infusion Pump Instructions For Use Manual."`) and numbered
procedure-list items that are grammatically full sentences, not headings
(`"5. Close the Roller Clamp and the Pinch Clamp."`). Real headings end with nothing,
or at most a colon (`WARNING:`) — never a full stop.

**(e) Numbered-prefix matching now requires an actual period present** (either from a
multi-level `.digit` group or a standalone trailing period), not just digits followed
by whitespace. This rejects bare page-number-prefixed running footers (`"10 Z-800F
Instructions for Use"`) from being treated as numbered section headings at all.

## 3. Empirical Validation (performed before finalizing this design, not after)

Ran the full classifier (a-e) against the real extracted text of all 7 documents,
built actual `Section` objects, and measured chunk-count impact — not just raw
heading-line counts.

**Primary goal confirmed fixed:** largest single section, as % of the document's
chunks, dropped from 94-100% to 1-10% across `78369`/`188844`/`153781` — matching or
beating the 73141/Z-800F/intera-3000/FreedomEdge documents that were already fine.

**Regression found and addressed separately (§4), not by further classifier changes:**
Title-Case detection also fires on tabular/reference content (alarm-code tables,
accessory-list tables) in the IFU-heavy documents, producing many very short sections
— Z-800F went from 112 to 261 sections, with 24% of them under 60 characters;
intera-3000 (previously verified correct) picked up the same pattern at 29%. A further
detection-time guard was considered (reject a heading if its own body looks
table-cell-like) and explicitly rejected: it would be a third heuristic stacked on an
already twice-patched function, with the same regression risk pattern already seen
twice, requiring the same 7-document validation pass again for uncertain benefit.

## 4. Fix, Part 2: Chunk-Length Filter (separate mechanism, not a heading-detection change)

Rather than trying to make `detect_sections` distinguish "real heading" from "table
row" perfectly (a hard problem without PDF layout information), filter at the chunk
level instead, in `chunk_pdf_text`: **drop any produced chunk whose text is under 40
characters.**

This is deliberately a different question than the one that caused the intera-3000
regression risk earlier ("is this heading real") — it's "is this chunk worth
indexing," decided after chunking, not during section detection. It does not touch
`detect_sections` further.

**Threshold chosen empirically**, not as a round number: built actual chunks (via the
real recursive splitter) for all 7 documents (950 chunks total) and inspected every
chunk under 150 characters by hand. Findings that fixed the threshold at 40:

- A genuine safety warning exists at 83 characters (`CAUTION: → "Do not continuously
  use the Zyno Administration set in the pump more than 72 hours."`), plus other real
  content in the 55-92 character range (a flow-rate spec, procedural instructions).
  These must survive.
- Everything found under 40 characters is bare numbers, single-word/short-phrase table
  cells, or truncated codes (`"Low"`, `"SN"`, `"6612B Secondary Red"`) — no real
  warnings or instructions found below this line.
- One accepted minor tradeoff: a 38-character symbol-glossary caption (`"Keep dry" →
  "Nonpyrogenic, see instructions for use"`) is lost — low individual retrieval value,
  judged acceptable.
- At threshold=40: 77 of 950 chunks (8.1%) dropped, all confirmed low-value on manual
  inspection.

**Why chunk-level filtering, not detection-time suppression:** this project's
retrieval is hybrid (BM25 + dense, per design doc §4). BM25's length normalization
favors short, keyword-dense documents — a near-empty chunk containing just an alarm
code or reference number is a real candidate to rank highly for an exact-match query
via the sparse leg specifically (not just "unlikely to be retrieved" in general),
then leave generation with nothing useful. Filtering these out is a retrieval-quality
fix, not cosmetic.

**Known, accepted residual limitation (magnitude corrected after real-corpus
validation — see below):** this does not eliminate all noise — repeated page-footer
fragments and table-header-row repeats in the 40-90 character range remain in the
corpus (e.g. `"...Z-800F Instructions for Use \nP/N 800F-IFU-2602, Rev. O"` appears
many times, each ~58 characters). These are real English text, not the
bare-code/single-word pattern the 40-character filter targets, and are judged lower
risk for the sparse-retrieval-favors-short-documents failure mode this fix is
specifically closing.

**Real-corpus re-validation (Task 3 of the implementation plan) found this residual
is substantially larger, on 2 of 7 documents specifically, than the paragraph above
originally estimated:**

| Document | Top repeated label | Share of document's chunks |
|---|---|---|
| `188844` | `Contains Nonbinding Recommendations` | **40%** |
| `153781` | `Contains Nonbinding Recommendations` | **41%** |
| all other 5 documents | (real section names) | 5-12% |

This is not the original swallowing bug (no document collapses to one catch-all
section anymore) — it is the repeated running-footer phrase reappearing as the
"current heading" across many separate section instances (13 for `188844`, 25 for
`153781`) that happen to share the same uninformative label, rather than one
continuous blob.

**A targeted fix (exact-duplicate-heading-text suppression: track heading strings
already accepted, treat an exact repeat as continuation of the current section
rather than a new boundary) was designed and empirically checked against the same
corpus before implementation, per this project's established practice of validating
before committing to a threshold — and rejected.** The check required: (1) confirm no
document has a legitimate heading intentionally repeated across genuinely different
content (the risk the design anticipated), and (2) confirm the repeated `Contains
Nonbinding Recommendations` instances are predominantly thin (safe to merge). Check
(1) found a real counter-example: `153781`'s `Outcome: Basic Documentation Level`
repeats 4 times, each instance a genuinely different worked example (a blood pressure
monitor, a behavioral therapy device, an OTC heart-rhythm app, an in-vitro nucleic
acid test device — 1,088 to 4,308 characters each, all substantial). Blanket
exact-duplicate suppression would have merged these 4 distinct examples into one
section, losing the ability to cite one specifically — a new regression, not a fix.

A refined "thinness gate" (suppress a repeat only if the same heading's *prior*
instance was itself thin) was considered as a middle path and checked empirically
against both known cases before implementation, not assumed to work:

- The `Outcome: Basic/Enhanced Documentation Level` instances (which must never be
  suppressed) are all 1,088-4,308 characters — safely classifiable as "substantial"
  under any threshold set below ~1,000 characters.
- But `Contains Nonbinding Recommendations` is **not** predominantly thin under such
  a threshold: for `188844`, 2 of its 13 instances (18,645 and 22,992 characters)
  account for 75% of all content attributed to that label; for `153781`, 5 of 25
  instances account for 65%. A threshold low enough to protect the Documentation
  Level examples leaves nearly all of the `Contains Nonbinding Recommendations`
  content classified as "substantial" too — the signal cannot separate "boilerplate
  repeat" from "real content that inherited a stale repeated label because no further
  internal heading was detected before the next real heading." That's a materially
  different problem (missing sub-heading detection within long spans) than
  duplicate-heading merging can address, on this corpus.

**Decision: no further detection-time change.** This is the third distinct heuristic
considered for this function (after the Title-Case classifier itself and the
masthead-suppression scoping) to fail a clean empirical fit; per this project's own
established discipline (see Task 13's history in the retrieval-pipeline implementation
plan — three review rounds on one function is the point to stop, not keep patching),
continuing to stack heuristics on `detect_sections` has hit diminishing returns. The
40-41% figure on `188844`/`153781` stands as a known, larger-than-originally-estimated
limitation, not addressed in this design. It may be worth revisiting as a *generation*
problem (does the model, given a chunk whose only section label is
`Contains Nonbinding Recommendations`, still answer correctly from the chunk's actual
text — the label is metadata, not the retrieved content itself) once Phase 3's
citation-grounding work is designed, rather than reopening this function a fourth time.

## 5. Scope

**In scope:** `src/fda_device_rag/chunking/sections.py` (`_is_heading` /
`HEADING_PATTERN` replacement per §2), `src/fda_device_rag/chunking/pdf_chunker.py`
(minimum-chunk-length filter per §4).

**Out of scope, explicitly:** further heading-detection refinement for table/list
content (rejected in §3); multi-column PDF layout handling (investigated and retracted
earlier in this cycle — confirmed not a real bug on the actual corpus); merging or
otherwise combining dropped-chunk content into neighboring chunks (dropping is
sufficient per the empirical review; merging adds complexity with no identified need).

## 6. Testing

Existing `tests/chunking/test_sections.py` tests (ALL-CAPS, single- and multi-level
numbered, no-headings-fallback, empty-section-dropping, consecutive-same-heading
merge) must continue passing unchanged. New tests required:

- Title-Case heading detected (`"Preface"`, `"Table of Contents"`).
- Title-Case candidate with a long lowercase word rejected (ordinary prose line).
- Catalog-code/table-row line rejected (real-word guard) — both ALL-CAPS and
  Title-Case shapes.
- Trailing-period line rejected (wrapped fragment / numbered list item), including
  after a numbered prefix.
- Bare page-number prefix (no period) does NOT count as a numbered heading.
- Masthead suppression: single short ALL-CAPS token before any real heading is
  suppressed; the same token occurring after a real heading has already fired is
  NOT suppressed (regression guard for the FreedomEdge `FLANGE` case).
- Chunk-length filter: a section producing a sub-40-character chunk is dropped from
  `chunk_pdf_text`'s output; a chunk at or above 40 characters is kept (boundary
  test at exactly 40).

No changes to `Section`/`Chunk`/`ChunkMetadata` interfaces — this is an internal
implementation change to two existing functions, not a new module or a signature
change. Nothing downstream (Task 9-13's embedder, stores, retriever) needs to change.
