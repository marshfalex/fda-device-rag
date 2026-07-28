import re

from rank_bm25 import BM25Okapi

from fda_device_rag.models import Chunk, ScoredChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        tokenized_corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def query(self, query_text: str, top_k: int = 20) -> list[ScoredChunk]:
        tokenized_query = tokenize(query_text)
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            ScoredChunk(
                id=self._chunks[i].id,
                score=float(scores[i]),
                text=self._chunks[i].text,
                metadata=self._chunks[i].metadata,
            )
            for i in ranked_indices
        ]
