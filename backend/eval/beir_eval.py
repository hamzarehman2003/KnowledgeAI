"""Score retrieval quality on a BEIR dataset and sweep the distance cutoff.

Primary metric is nDCG@10. NFCorpus judgements are graded (1 = relevant,
2 = highly relevant) and very dense -- a median of 16 relevant documents per
query -- so recall@5 is capped near 30% even for a perfect retriever and would
read as failure. nDCG@10 handles both facts correctly.
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import chromadb

from app.core.config import get_settings
from app.rag.embeddings import OllamaEmbeddingService

# Fetched per query: deep enough for recall@100 and a meaningful threshold sweep.
CANDIDATE_DEPTH = 100
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Read a BEIR qrels TSV into {query_id: {doc_id: grade}}."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        next(handle)  # header: query-id, corpus-id, score
        for line in handle:
            if not line.strip():
                continue
            query_id, doc_id, score = line.strip().split("\t")
            grade = int(score)
            if grade > 0:
                qrels[query_id][doc_id] = grade
    return dict(qrels)


def load_queries(path: Path, wanted: set[str]) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["_id"] in wanted:
                queries[record["_id"]] = record["text"]
    return queries


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Exponential-gain nDCG, matching the pytrec_eval convention BEIR reports."""
    gains = [(2 ** relevance.get(doc_id, 0)) - 1 for doc_id in ranked_ids[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(((2**grade) - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    if not relevance:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevance)
    return hits / len(relevance)


def reciprocal_rank(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevance:
            return 1.0 / rank
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--collection", default="nfcorpus_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="score only N queries")
    arguments = parser.parse_args()

    settings = get_settings()
    qrels = load_qrels(arguments.dataset_dir / "qrels" / f"{arguments.split}.tsv")
    queries = load_queries(arguments.dataset_dir / "queries.jsonl", set(qrels))
    query_ids = sorted(queries)
    if arguments.limit:
        query_ids = query_ids[: arguments.limit]
    print(f"scoring {len(query_ids)} '{arguments.split}' queries against {arguments.collection!r}\n")

    embedder = OllamaEmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    collection = client.get_collection(name=arguments.collection)

    started = time.perf_counter()
    vectors = embedder.embed_texts([queries[query_id] for query_id in query_ids])
    print(f"embedded {len(vectors)} queries in {time.perf_counter() - started:.1f}s")

    result = collection.query(
        query_embeddings=[list(vector) for vector in vectors],
        n_results=CANDIDATE_DEPTH,
        include=["distances"],
    )
    all_ids = result["ids"]
    all_distances = result["distances"]

    metrics = {"nDCG@10": [], "recall@10": [], "recall@100": [], "MRR@10": []}
    for index, query_id in enumerate(query_ids):
        ranked = all_ids[index]
        relevance = qrels[query_id]
        metrics["nDCG@10"].append(ndcg_at_k(ranked, relevance, 10))
        metrics["recall@10"].append(recall_at_k(ranked, relevance, 10))
        metrics["recall@100"].append(recall_at_k(ranked, relevance, 100))
        metrics["MRR@10"].append(reciprocal_rank(ranked, relevance, 10))

    print(f"\n{'metric':12} {'score':>7}")
    print("-" * 20)
    for name, values in metrics.items():
        print(f"{name:12} {sum(values)/len(values):7.4f}")

    # --- distance cutoff sweep -------------------------------------------------
    # Chooses the threshold that discards irrelevant chunks before they reach the
    # model, without silently refusing questions the corpus can actually answer.
    print(f"\n{'cutoff':>7} {'kept/q':>8} {'precision':>10} {'recall@100':>11} {'empty':>7}")
    print("-" * 48)
    for threshold in THRESHOLDS:
        kept_counts, precisions, recalls, empties = [], [], [], 0
        for index, query_id in enumerate(query_ids):
            relevance = qrels[query_id]
            kept = [
                doc_id
                for doc_id, distance in zip(all_ids[index], all_distances[index])
                if distance <= threshold
            ]
            kept_counts.append(len(kept))
            if not kept:
                empties += 1
                precisions.append(0.0)
            else:
                precisions.append(sum(1 for d in kept if d in relevance) / len(kept))
            recalls.append(recall_at_k(kept, relevance, CANDIDATE_DEPTH))
        n = len(query_ids)
        print(
            f"{threshold:7.2f} {sum(kept_counts)/n:8.1f} {sum(precisions)/n:10.3f} "
            f"{sum(recalls)/n:11.3f} {empties/n:6.0%}"
        )


if __name__ == "__main__":
    main()
