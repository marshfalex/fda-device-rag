"""Deterministic citation extraction from generated answer text -- reused
verbatim from
docs/superpowers/specs/2026-07-28-fda-device-rag-architecture-design.md
section 5, adapted to the {number: ScoredChunk} map build_prompt produces.
Regex-parses bracket-numbered citations the model wrote inline, rather than
trusting a self-reported citation list -- an 8B model can fabricate a
citations array as easily as it can fabricate an answer; parsing numbers it
already wrote inline is citation-by-what-it-did, not citation-by-what-it-
claims."""
import re

from fda_device_rag.models import ScoredChunk


def extract_citations(answer_text: str, chunk_map: dict[int, ScoredChunk]) -> tuple[list[ScoredChunk], str]:
    cited_nums = {int(n) for n in re.findall(r'\[(\d)\]', answer_text)}
    citations = [chunk_map[n] for n in cited_nums if n in chunk_map]
    if not citations:
        # The model sometimes skips the bracket format entirely -- fall back
        # to showing all context provided, honestly labeled as unverified.
        citations = list(chunk_map.values())
        citation_label = "context provided (model did not cite specific passages)"
    else:
        citation_label = "cited by the model"
    return citations, citation_label
