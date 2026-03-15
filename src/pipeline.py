# src/pipeline.py
from ingestion import DocumentIngester
from chunking import DocumentChunker
from vector_store import VectorStore
from generator import RAGGenerator
from typing import Optional


class RAGPipeline:
    """
    Single entry point for the full RAG system.
    Composes ingestion → chunking → vector store → generation.
    """

    def __init__(self):
        print("=== Initialising RAGStack ===")
        self.ingester = DocumentIngester()
        self.chunker = DocumentChunker()
        self.store = VectorStore()
        self.generator = RAGGenerator()

    def index(self, urls: list[str] = None) -> None:
        """
        Ingest + chunk + embed documents into the vector store.
        Call this once when adding new documents.
        """
        docs = self.ingester.ingest(urls=urls)
        chunks = self.chunker.chunk_documents(docs)
        self.store.add_chunks(chunks)
        print(f"  Indexed {self.store.count()} chunks total")

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Ask a question. Returns answer + sources.
        """
        chunks = self.store.retrieve(question, top_k=top_k)
        result = self.generator.generate(question, chunks)
        return result

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