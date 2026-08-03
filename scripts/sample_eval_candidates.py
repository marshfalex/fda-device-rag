# scripts/sample_eval_candidates.py
"""Runs the deterministic candidate-sampling pipeline (recall/MAUDE
categories + PDF section instances) and writes the intermediate,
uncommitted candidates file consumed by the collaborative question-phrasing
session (design doc docs/superpowers/specs/2026-08-03-eval-harness-design.md,
section 8, step 1). This is NOT the frozen benchmark -- see
data/eval/questions.json for that, and note data/eval/candidates.json is
gitignored specifically so it can never be committed in its place.

Usage: python scripts/sample_eval_candidates.py [--seed SEED]
"""
import argparse
import json
from pathlib import Path

from fda_device_rag.documents.pdf_document import extract_pdf_text
from fda_device_rag.chunking.pdf_chunker import chunk_pdf_text
from fda_device_rag.eval.sampling import (
    check_section_diversity,
    sample_recall_candidates,
    sample_event_candidates,
    sample_pdf_section_candidates,
)

DATA_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/eval/candidates.json")
DEFAULT_SEED = "fda-device-rag-eval-2026-08-03"

GUIDANCE_DOCS = ["78369", "188844", "153781", "73141"]
GUIDANCE_QUOTA_PER_DOC = 3
IFU_DOCS = ["Z-800F_Instructions_for_Use_Rev_O", "intera-3000-pump-ifu", "FreedomEdge_Domestic_IFU_347201_Rev_B"]
IFU_QUOTA_PER_DOC = 4


def _read_pull_date() -> str:
    pull_date_path = DATA_DIR / "pull_date.txt"
    return pull_date_path.read_text().strip() if pull_date_path.exists() else ""


def _load_pdf_chunks(document_title: str, subdir: str, source_type: str, retrieved_date: str):
    pdf_path = DATA_DIR / subdir / f"{document_title}.pdf"
    text = extract_pdf_text(pdf_path)
    return chunk_pdf_text(
        text,
        source_type=source_type,
        source_url=str(pdf_path),
        document_title=document_title,
        retrieved_date=retrieved_date,
        id_prefix=document_title,
    )


def _sample_pdf_group(docs, subdir, source_type, benchmark_source_type, quota_per_doc, retrieved_date, seed):
    candidates = []
    skip_log = []
    for doc in docs:
        chunks = _load_pdf_chunks(doc, subdir, source_type, retrieved_date)
        selected, skips = sample_pdf_section_candidates(doc, chunks, quota_per_doc, base_seed=seed)
        skip_log.extend(skips)
        for inst in selected:
            candidates.append({
                "source_type": benchmark_source_type,
                "document_title": inst.document_title,
                "section_name": inst.section_name,
                "instance_ordinal": inst.instance_ordinal,
                "text": inst.text,
            })
    return candidates, skip_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()

    retrieved_date = _read_pull_date()
    recalls = json.loads((DATA_DIR / "recalls.json").read_text())
    events = json.loads((DATA_DIR / "events.json").read_text())

    candidates = []
    candidates.extend(sample_recall_candidates(recalls, retrieved_date=retrieved_date, base_seed=args.seed))
    candidates.extend(sample_event_candidates(events, retrieved_date=retrieved_date, base_seed=args.seed))

    guidance_candidates, guidance_skips = _sample_pdf_group(
        GUIDANCE_DOCS, "guidance_pdfs", "guidance_pdf", "guidance", GUIDANCE_QUOTA_PER_DOC, retrieved_date, args.seed,
    )
    ifu_candidates, ifu_skips = _sample_pdf_group(
        IFU_DOCS, "ifu_pdfs", "ifu_pdf", "ifu", IFU_QUOTA_PER_DOC, retrieved_date, args.seed,
    )
    # Design doc section 8 step 2: verify no two candidates within the same
    # document's quota share a section_name label. Runs across guidance and
    # IFU candidates together, and before any output is written, so a
    # violation is a hard error rather than a silently-shipped candidates
    # file.
    check_section_diversity(guidance_candidates + ifu_candidates)

    candidates.extend(guidance_candidates)
    candidates.extend(ifu_candidates)
    skip_log = guidance_skips + ifu_skips

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"seed": args.seed, "candidates": candidates, "furniture_skips": skip_log}, indent=2))

    print(f"Sampled {len(candidates)} candidates (seed={args.seed!r}) -> {OUTPUT_PATH}")
    print(f"Furniture skips: {len(skip_log)}")


if __name__ == "__main__":
    main()
