# src/vector_store.py
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


class VectorStore:
    """
    Wraps ChromaDB with Ollama embeddings.
    Handles storing chunks and retrieving relevant ones by query.
    """

    def __init__(
        self,
        collection_name: str = "ragstack",
        persist_dir: str = "data/chroma",
        embedding_model: str = "nomic-embed-text",  # pulled earlier via ollama
        top_k: int = 5,
    ):
        self.collection_name = collection_name
        self.persist_dir = str(Path(persist_dir))
        self.top_k = top_k

        print(f"  Loading embedding model: {embedding_model}")
        self.embeddings = OllamaEmbeddings(model=embedding_model)

        # Persistent — survives restarts, no re-embedding needed
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def add_chunks(self, chunks: List[Document]) -> None:
        """
        Embed and store chunks in ChromaDB.
        Skips if collection already has data — avoids duplicate embeddings
        on re-runs.
        """
        existing = self.db._collection.count()

        if existing > 0:
            print(f"  Collection already has {existing} chunks — skipping embed.")
            print(f"  Delete data/chroma/ to force re-embed.")
            return

        print(f"  Embedding {len(chunks)} chunks (this may take a minute)...")
        self.db.add_documents(chunks)
        print(f"  Stored {len(chunks)} chunks in ChromaDB at {self.persist_dir}")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Embed the query and return the top-K most similar chunks.
        """
        k = top_k or self.top_k
        results = self.db.similarity_search(query, k=k)
        return results

    def retrieve_with_scores(
        self, query: str, top_k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """
        Same as retrieve() but also returns similarity scores.
        Useful for debugging retrieval quality.
        Score is cosine distance — lower = more similar.
        """
        k = top_k or self.top_k
        results = self.db.similarity_search_with_score(query, k=k)
        return results

    def count(self) -> int:
        """How many chunks are currently stored."""
        return self.db._collection.count()


# Quick test
if __name__ == "__main__":
    from ingestion import DocumentIngester
    from chunking import DocumentChunker

    # Step 1 — ingest
    ingester = DocumentIngester()
    docs = ingester.ingest(
        urls=["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
    )

    # Step 2 — chunk
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(docs)

    # Step 3 — embed + store
    print("\n=== Vector Store ===")
    store = VectorStore()
    store.add_chunks(chunks)
    print(f"  Total chunks in DB: {store.count()}")

    # Step 4 — test retrieval
    query = "How is RAG evaluated?"
    print(f"\nQuery: '{query}'")
    results = store.retrieve_with_scores(query, top_k=3)

    for i, (doc, score) in enumerate(results):
        print(f"\n  Result {i+1} (score: {score:.4f})")
        print(f"  Source: {doc.metadata.get('source', 'unknown')}")
        print(f"  Text: {doc.page_content[:150]}...")

# Quick test — replace the if __name__ block temporarily
# if __name__ == "__main__":
#     import shutil
    
#     # Find and delete wherever chroma is actually storing data
#     store = VectorStore()
#     print(f"Chroma storing at: {store.db._client._system.settings.persist_directory}")
#     store.db.delete_collection()
#     print("Collection deleted!")