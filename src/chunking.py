# src/chunking.py
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


class DocumentChunker:
    """
    Splits documents into overlapping chunks sized by token count.
    Uses RecursiveCharacterTextSplitter which tries to split on
    natural boundaries: paragraphs → sentences → words → characters.
    """

    def __init__(
        self,
        chunk_size: int = 600,       # target tokens per chunk
        chunk_overlap: int = 100,    # overlap tokens between chunks
        model_name: str = "gpt2",    # tokenizer — gpt2 is free, no API needed
        min_chunk_length: int = 100,  # discard junk chunks (nav menus, headers)
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

        # tiktoken counts real tokens — more accurate than character splitting
        self.tokenizer = tiktoken.get_encoding(model_name)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_count,  # split by tokens, not chars
            separators=["\n\n", "\n", ". ", " ", ""],  # natural boundaries first
        )

    def _token_count(self, text: str) -> int:
        """Count tokens in a string using tiktoken."""
        return len(self.tokenizer.encode(text))

    def _is_valid_chunk(self, text: str) -> bool:
        """
        Filter out garbage chunks — nav menus, single words,
        whitespace-heavy content scraped from web pages.
        """
        stripped = text.strip()
        if len(stripped) < self.min_chunk_length:
            return False
        # Discard chunks that are mostly newlines (web nav junk)
        newline_ratio = stripped.count("\n") / max(len(stripped), 1)
        if newline_ratio > 0.3:
            return False
        #Wikipedia-specific footer junk
        if stripped.startswith("Category") or stripped.startswith("Retrieved from"):
            return False
        #chunks with very few words are unlikely to be useful
        word_count = len(stripped.split())
        if word_count < 20:
            return False
        if stripped.startswith("Jump to content"):
            return False
        if stripped.startswith("vte"):
            return False
        return True

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Main entry point. Takes raw documents, returns clean chunks
        with inherited + enriched metadata.
        """
        print("\n=== Chunking ===")
        all_chunks = []

        for doc in documents:
            raw_chunks = self.splitter.split_documents([doc])

            # Filter junk and enrich metadata
            for i, chunk in enumerate(raw_chunks):
                if not self._is_valid_chunk(chunk.page_content):
                    continue

                # Carry forward parent metadata + add chunk-specific info
                chunk.metadata.update({
                    "chunk_index": i,
                    "token_count": self._token_count(chunk.page_content),
                    "source": doc.metadata.get("source", "unknown"),
                })
                all_chunks.append(chunk)

        print(f"  Input documents : {len(documents)}")
        print(f"  Output chunks   : {len(all_chunks)}")
        if all_chunks:
            token_counts = [c.metadata["token_count"] for c in all_chunks]
            print(f"  Avg chunk size  : {sum(token_counts)//len(token_counts)} tokens")
            print(f"  Min / Max       : {min(token_counts)} / {max(token_counts)} tokens")

        return all_chunks


# Quick test
if __name__ == "__main__":
    from ingestion import DocumentIngester

    ingester = DocumentIngester()
    docs = ingester.ingest(
        urls=["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
    )

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(docs)

    print(f"\nSample chunk:")
    if chunks:
        print(f"  Tokens : {chunks[5].metadata['token_count']}")
        print(f"  Source : {chunks[5].metadata['source']}")
        print(f"  Text   : {chunks[5].page_content[:200]}...")