# src/pipeline.py
from ingestion import DocumentIngester
from chunking import DocumentChunker
from vector_store import VectorStore
from generator import RAGGenerator
from typing import Optional
from retriever import HybridRetriever, ReRanker
from config import CONFIG


class RAGPipeline:
    """
    Single entry point for the full RAG system.
    Composes ingestion → chunking → vector store → generation.
    """

    def __init__(self):
        print("=== Initialising RAGStack ===")
        cfg = CONFIG

        self.ingester = DocumentIngester(
        )
        self.chunker = DocumentChunker(
            chunk_size=cfg["chunking"]["chunk_size"],
            chunk_overlap=cfg["chunking"]["chunk_overlap"],
            min_chunk_length=cfg["chunking"]["min_chunk_length"],
        )
        self.store = VectorStore(
            collection_name=cfg["vector_store"]["collection_name"],
            persist_dir=cfg["vector_store"]["persist_dir"],
            embedding_model=cfg["embeddings"]["model"],
            top_k=cfg["retrieval"]["top_k"],
        )
        self.generator = RAGGenerator(
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
            prompt_template=cfg["prompts"]["rag_prompt"],
        )
        self.reranker = ReRanker(
            model=cfg["reranker"]["model"]
        ) if cfg["reranker"]["enabled"] else None

        self.chunks = []  # kept in memory for hybrid retrieval

    def index(self, urls: list[str] = None) -> None:
        """
        Ingest + chunk + embed documents into the vector store.
        Call this once when adding new documents.
        """
        docs = self.ingester.ingest(urls=urls)
        self.chunks = self.chunker.chunk_documents(docs)
        self.store.add_chunks(self.chunks)

        #Always load from db do chunks are available even when skipping embed
        self.chunks = self.store.get_all_chunks()
        print(f"  Indexed {self.store.count()} chunks total")

    def query(self, question: str) -> dict:
        """Ask a question, get a grounded answer back."""
        cfg = CONFIG["retrieval"]

        # Step 1 — hybrid retrieval (wide net)
        retriever = HybridRetriever(
            vector_db=self.store.db,
            chunks=self.chunks,
            bm25_weight=cfg["bm25_weight"],
            vector_weight=cfg["vector_weight"],
        )
        candidates = retriever.retrieve(question, top_k=cfg["candidates"])

        # Step 2 — re-rank if enabled (precision filter)
        if self.reranker:
            chunks = self.reranker.rerank(
                question, candidates, top_k=cfg["top_k"]
            )
        else:
            chunks = candidates[:cfg["top_k"]]

        # Step 3 — generate grounded answer
        return self.generator.generate(question, chunks)

    def ask(self, question: str) -> None:
        """
        Pretty-print a question and answer.
        Convenience method for interactive use.
        """
        print(f"\nQ: {question}")
        result = self.query(question)
        print(f"A: {result['answer']}")
        print(f"   Sources: {result['sources']}")
        print(f"   Chunks used: {result['num_chunks']}")


if __name__ == "__main__":
    rag = RAGPipeline()

    # Index documents (skips if already embedded)
    rag.index(
        urls=["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
    )

    # Ask questions
    rag.ask("How is RAG evaluated?")
    rag.ask("What are the limitations of RAG?")
    rag.ask("What is the capital of France?")