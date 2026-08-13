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
from app.rag.reranking import CrossEncoderReranker
from app.rag.vector_store import RetrievedChunk
from eval.beir_index import load_corpus
from eval.bm25 import BM25Index, reciprocal_rank_fusion

# Fetched per query: deep enough for recall@100 and a meaningful threshold sweep.
CANDIDATE_DEPTH = 100
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
# RRF's k damps how much the very top ranks dominate; 60 is the published default.
RRF_K_VALUES = [5, 10, 20, 40, 60, 100]


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


def rerank_candidates(
    reranker: CrossEncoderReranker,
    *,
    question: str,
    ranked_ids: list[str],
    documents: list[str],
    distances: list[float],
    depth: int,
) -> list[str]:
    """Reorder the head of one query's candidate list with a cross-encoder.

    Only the first ``depth`` candidates are rescored; the tail keeps its
    embedding order so deeper metrics like recall@100 stay comparable.
    """
    head = [
        RetrievedChunk(
            document_id=doc_id,
            page_number=0,
            chunk_index=0,
            text=text,
            distance=distance,
        )
        for doc_id, text, distance in zip(
            ranked_ids[:depth], documents[:depth], distances[:depth], strict=True
        )
    ]
    reranked = reranker.rerank(question=question, chunks=head)
    return [candidate.chunk.document_id for candidate in reranked] + ranked_ids[depth:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--collection", default="nfcorpus_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="score only N queries")
    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help="dense embeddings, lexical BM25, or RRF fusion of both",
    )
    # 10 rather than the conventional 60: chosen on the dev split, where 5-20 all
    # beat 60, then reported on test. Tuning this on test would be overfitting.
    parser.add_argument("--rrf-k", type=int, default=10, help="RRF damping constant")
    parser.add_argument("--sweep-rrf-k", action="store_true", help="score every RRF_K_VALUES setting")
    parser.add_argument("--rerank", action="store_true", help="rescore candidates with a cross-encoder")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--rerank-depth", type=int, default=50, help="candidates to rescore per query")
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    arguments = parser.parse_args()

    settings = get_settings()
    qrels = load_qrels(arguments.dataset_dir / "qrels" / f"{arguments.split}.tsv")
    queries = load_queries(arguments.dataset_dir / "queries.jsonl", set(qrels))
    query_ids = sorted(queries)
    if arguments.limit:
        query_ids = query_ids[: arguments.limit]
    print(
        f"scoring {len(query_ids)} '{arguments.split}' queries "
        f"[mode={arguments.mode}{', rerank' if arguments.rerank else ''}]\n"
    )

    if arguments.rerank and arguments.mode != "dense":
        parser.error("--rerank needs candidate text, which only --mode dense supplies")

    dense_ids: list[list[str]] = []
    all_distances: list[list[float]] | None = None
    result: dict = {}

    if arguments.mode in {"dense", "hybrid"}:
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
            # Documents are needed only to feed the cross-encoder, which reads the
            # question and passage text together rather than their embeddings.
            include=["distances", "documents"] if arguments.rerank else ["distances"],
        )
        dense_ids = result["ids"]
        all_distances = result["distances"]

    bm25_ids: list[list[str]] = []
    if arguments.mode in {"bm25", "hybrid"}:
        started = time.perf_counter()
        corpus = load_corpus(arguments.dataset_dir, None)
        index = BM25Index(
            [document["id"] for document in corpus],
            [document["text"] for document in corpus],
        )
        bm25_ids = [
            index.search(queries[query_id], CANDIDATE_DEPTH) for query_id in query_ids
        ]
        print(
            f"BM25 indexed {len(corpus)} docs and ranked {len(query_ids)} queries "
            f"in {time.perf_counter() - started:.1f}s"
        )

    if arguments.mode == "dense":
        all_ids = dense_ids
    elif arguments.mode == "bm25":
        all_ids = bm25_ids
        all_distances = None
    else:
        all_ids = [
            reciprocal_rank_fusion([dense, lexical], k=arguments.rrf_k, top_k=CANDIDATE_DEPTH)
            for dense, lexical in zip(dense_ids, bm25_ids, strict=True)
        ]
        # Fusion reorders ids away from the distances they came with.
        all_distances = None

    if arguments.rerank:
        reranker = CrossEncoderReranker(
            model_name=arguments.rerank_model,
            batch_size=arguments.rerank_batch_size,
        )
        all_documents = result["documents"]
        started = time.perf_counter()
        all_ids = [
            rerank_candidates(
                reranker,
                question=queries[query_id],
                ranked_ids=all_ids[index],
                documents=all_documents[index],
                distances=all_distances[index],
                depth=min(arguments.rerank_depth, CANDIDATE_DEPTH),
            )
            for index, query_id in enumerate(query_ids)
        ]
        elapsed = time.perf_counter() - started
        print(
            f"reranked top-{arguments.rerank_depth} of {len(query_ids)} queries "
            f"with {arguments.rerank_model} in {elapsed:.1f}s "
            f"({elapsed / len(query_ids) * 1000:.0f} ms/query)"
        )
        # The sweep below reads distances positionally, so it would silently
        # mispair them with the reordered ids.
        all_distances = None

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

    # --- RRF damping sweep -----------------------------------------------------
    if arguments.mode == "hybrid" and arguments.sweep_rrf_k:
        print(f"\n{'rrf_k':>7} {'nDCG@10':>9} {'recall@10':>11} {'recall@100':>12} {'MRR@10':>8}")
        print("-" * 50)
        for k in RRF_K_VALUES:
            fused = [
                reciprocal_rank_fusion([dense, lexical], k=k, top_k=CANDIDATE_DEPTH)
                for dense, lexical in zip(dense_ids, bm25_ids, strict=True)
            ]
            scores = {
                "ndcg": [ndcg_at_k(fused[i], qrels[q], 10) for i, q in enumerate(query_ids)],
                "r10": [recall_at_k(fused[i], qrels[q], 10) for i, q in enumerate(query_ids)],
                "r100": [recall_at_k(fused[i], qrels[q], 100) for i, q in enumerate(query_ids)],
                "mrr": [reciprocal_rank(fused[i], qrels[q], 10) for i, q in enumerate(query_ids)],
            }
            n = len(query_ids)
            print(
                f"{k:7d} {sum(scores['ndcg'])/n:9.4f} {sum(scores['r10'])/n:11.4f} "
                f"{sum(scores['r100'])/n:12.4f} {sum(scores['mrr'])/n:8.4f}"
            )

    # --- distance cutoff sweep -------------------------------------------------
    # Chooses the threshold that discards irrelevant chunks before they reach the
    # model, without silently refusing questions the corpus can actually answer.
    if all_distances is None:
        print("\nskipping cutoff sweep: this mode reorders ids away from their distances")
        return

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
