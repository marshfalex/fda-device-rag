# scripts/upload_corpus_to_hf.py
"""One-time (or re-run-when-the-corpus-changes) upload of the pre-built
retrieval index to a Hugging Face Hub Dataset repo, so the deployed demo
(app.py) can download it at startup instead of rebuilding against openFDA
live on every cold start.

Uploads only what load_retrieval_stack() actually needs at query time --
data/chroma/ and data/bm25_index.pkl -- not data/raw/, which is only
needed to rebuild the index from scratch, not to query an already-built
one.

Requires an authenticated `hf` CLI session (run `hf auth login`, or set
the HF_TOKEN environment variable, before running this script) -- this
script never accepts a token as a command-line argument or prompts for
one, so a credential is never visible in shell history or this script's
output.

Usage: python scripts/upload_corpus_to_hf.py [--repo-id REPO_ID]
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "marshfalex/fda-device-rag-corpus"
CHROMA_DIR = Path("data/chroma")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = parser.parse_args()

    if not CHROMA_DIR.exists() or not BM25_INDEX_PATH.exists():
        print(f"Index not found ({CHROMA_DIR}, {BM25_INDEX_PATH}) -- run "
              f"scripts/pull_corpus.py then scripts/build_index.py first.")
        raise SystemExit(1)

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id, repo_type="dataset",
        folder_path=str(CHROMA_DIR), path_in_repo="chroma",
    )
    api.upload_file(
        repo_id=args.repo_id, repo_type="dataset",
        path_or_fileobj=str(BM25_INDEX_PATH), path_in_repo="bm25_index.pkl",
    )

    print(f"Uploaded {CHROMA_DIR} and {BM25_INDEX_PATH} -> "
          f"https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
