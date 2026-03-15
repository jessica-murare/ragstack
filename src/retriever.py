# src/retriever.py
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np


class HybridRetriever:
    """
    Combines BM25 keyword search with vector semantic search.
    Scores from both are normalised and merged using Reciprocal
    Rank Fusion (RRF) — a simple, robust combination strategy.
    """

    def __init__(
        self,
        vector_db: Chroma,
        chunks: List[Document],
        top_k: int = 5,
        bm25_weight: float = 0.4,    # 40% keyword
        vector_weight: float = 0.6,  # 60% semantic
    ):
        self.vector_db = vector_db
        self.chunks = chunks
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # Build BM25 index from chunk text
        tokenized = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, top_k: int) -> List[tuple[Document, float]]:
        """Keyword search using BM25. Returns (doc, score) pairs."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each chunk with its BM25 score
        scored = list(zip(self.chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _normalize(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Run both searches, normalize scores, combine with weights,
        return top-K deduplicated results.
        """
        k = top_k or self.top_k
        fetch_k = k * 2  # fetch more, then merge + trim

        # --- Vector search ---
        vector_results = self.vector_db.similarity_search_with_score(
            query, k=fetch_k
        )
        vector_docs = [doc for doc, _ in vector_results]
        vector_scores = [score for _, score in vector_results]

        # --- BM25 search ---
        bm25_results = self._bm25_search(query, fetch_k)
        bm25_docs = [doc for doc, _ in bm25_results]
        bm25_scores = [score for _, score in bm25_results]

        # --- Normalize both score lists ---
        norm_vector = self._normalize(vector_scores)
        norm_bm25 = self._normalize(bm25_scores)

        # --- Merge scores by content identity ---
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for doc, score in zip(vector_docs, norm_vector):
            key = doc.page_content[:100]  # use first 100 chars as ID
            # Vector distance = lower is better, so invert
            scores[key] = self.vector_weight * (1 - score)
            doc_map[key] = doc

        for doc, score in zip(bm25_docs, norm_bm25):
            key = doc.page_content[:100]
            bm25_contribution = self.bm25_weight * score
            if key in scores:
                scores[key] += bm25_contribution  # appeared in both — boost it
            else:
                scores[key] = bm25_contribution
                doc_map[key] = doc

        # --- Sort by combined score, return top-K ---
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked[:k]]
    

class ReRanker:
    """
    Cross-encoder re-ranker that scores each (query, chunk) pair
    together — much more precise than embedding-based similarity
    which scores query and chunks independently.

    Typical pipeline:
        1. Hybrid retrieval fetches top-20 candidates (broad net)
        2. Re-ranker scores all 20 against the query
        3. Return top-5 — now precision-ranked not just similarity-ranked
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"  Loading re-ranker: {model}")
        self.model = CrossEncoder(model)

    def rerank(
        self,
        query: str,
        chunks: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        Score each chunk against the query jointly,
        return top_k re-ranked results.
        """
        if not chunks:
            return []

        # Cross-encoder scores (query, chunk) pairs together
        pairs = [[query, chunk.page_content] for chunk in chunks]
        scores = self.model.predict(pairs)

        # Attach scores and sort
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Store score in metadata for transparency
        results = []
        for doc, score in scored[:top_k]:
            doc.metadata["rerank_score"] = round(float(score), 4)
            results.append(doc)

        return results    


# Quick test
if __name__ == "__main__":
    from ingestion import DocumentIngester
    from chunking import DocumentChunker
    from vector_store import VectorStore

    ingester = DocumentIngester()
    chunker = DocumentChunker()
    store = VectorStore()

    docs = ingester.ingest(
        urls=["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
    )
    chunks = chunker.chunk_documents(docs)
    store.add_chunks(chunks)

    query = "How is RAG evaluated?"

    # Step 1 — hybrid retrieval (cast wide net, fetch 8)
    print(f"\nQuery: '{query}'")
    retriever = HybridRetriever(store.db, chunks)
    candidates = retriever.retrieve(query, top_k=8)

    # Step 2 — re-rank (precision filter, keep top 3)
    reranker = ReRanker()
    final = reranker.rerank(query, candidates, top_k=3)

    print("\n--- After re-ranking (top 3) ---")
    for i, doc in enumerate(final, 1):
        print(f"\n  {i}. Score: {doc.metadata['rerank_score']}")
        print(f"     {doc.page_content[:150]}...")