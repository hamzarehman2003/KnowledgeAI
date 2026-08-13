"""Minimal BM25 ranking over an in-memory corpus.

Written out rather than pulled from a library so the scoring is inspectable and
the eval keeps no extra dependency. Correctness is checked against the published
BEIR BM25 baseline for NFCorpus (nDCG@10 ~0.32) rather than by unit tests alone.
"""

import math
import re
from collections import defaultdict

# Standard Robertson/Sparck-Jones parameters; BEIR's BM25 baselines use the same.
K1 = 0.9
B = 0.4


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; no stemming or stopword list."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Inverted-index BM25 so scoring touches only documents sharing a term."""

    def __init__(self, doc_ids: list[str], texts: list[str]) -> None:
        self.doc_ids = doc_ids
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for position, text in enumerate(texts):
            tokens = tokenize(text)
            self.doc_lengths.append(len(tokens))
            frequencies: dict[str, int] = defaultdict(int)
            for token in tokens:
                frequencies[token] += 1
            for token, frequency in frequencies.items():
                self.postings[token].append((position, frequency))

        self.doc_count = len(texts)
        self.average_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )
        self.inverse_document_frequency = {
            token: math.log(
                1 + (self.doc_count - len(entries) + 0.5) / (len(entries) + 0.5)
            )
            for token, entries in self.postings.items()
        }

    def search(self, query: str, top_k: int) -> list[str]:
        """Return the ids of the top_k documents, best first."""
        scores: dict[int, float] = defaultdict(float)
        for token in tokenize(query):
            entries = self.postings.get(token)
            if not entries:
                continue
            idf = self.inverse_document_frequency[token]
            for position, frequency in entries:
                length_norm = 1 - B + B * self.doc_lengths[position] / self.average_length
                scores[position] += idf * (frequency * (K1 + 1)) / (
                    frequency + K1 * length_norm
                )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [self.doc_ids[position] for position, _score in ranked[:top_k]]


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60, top_k: int) -> list[str]:
    """Merge ranked lists by rank position rather than by score.

    Dense cosine distances and BM25 scores live on different, corpus-dependent
    scales, so they cannot be added directly. RRF discards the scores and keeps
    only each document's position in each list.
    """
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] += 1.0 / (k + rank + 1)
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _score in ordered[:top_k]]
