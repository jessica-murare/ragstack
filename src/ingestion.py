from dotenv import load_dotenv
load_dotenv()
import os
import requests
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document


class DocumentIngester:
    """
    Loads documents from PDFs, Markdown files, or web URLs.
    Returns a flat list of LangChain Document objects.
    """

    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF file. Each page becomes one Document."""
        print(f"  Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # Tag each doc with source metadata
        for doc in docs:
            doc.metadata["source_type"] = "pdf"
        return docs

    def load_markdown(self, file_path: str) -> List[Document]:
        """Load a Markdown file as a single Document."""
        print(f"  Loading Markdown: {file_path}")
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = "markdown"
        return docs

    def load_web(self, url: str) -> List[Document]:
        """Scrape a web page and load its text content."""
        print(f"  Loading URL: {url}")
        loader = WebBaseLoader(url)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = "web"
            doc.metadata["url"] = url
        return docs

    def load_directory(self) -> List[Document]:
        """
        Scan data/raw/ and load all PDFs and Markdown files found.
        Returns all docs combined.
        """
        all_docs = []
        files = list(self.raw_data_dir.rglob("*"))

        if not files:
            print(f"  No files found in {self.raw_data_dir}")
            return all_docs

        for file_path in files:
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".pdf":
                    all_docs.extend(self.load_pdf(str(file_path)))
                elif suffix in (".md", ".markdown"):
                    all_docs.extend(self.load_markdown(str(file_path)))
                else:
                    # Skip unknown file types silently
                    pass
            except Exception as e:
                print(f"  WARNING: Failed to load {file_path}: {e}")

        return all_docs

    def load_urls(self, urls: List[str]) -> List[Document]:
        """Load a list of web URLs."""
        all_docs = []
        for url in urls:
            try:
                all_docs.extend(self.load_web(url))
            except Exception as e:
                print(f"  WARNING: Failed to load {url}: {e}")
        return all_docs

    def ingest(self, urls: List[str] = None) -> List[Document]:
        """
        Main entry point. Loads everything:
        - All PDFs + Markdown from data/raw/
        - Any URLs passed in
        """
        print("\n=== Document Ingestion ===")
        all_docs = []

        # Load local files
        local_docs = self.load_directory()
        all_docs.extend(local_docs)
        print(f"  Loaded {len(local_docs)} docs from local files")

        # Load URLs if provided
        if urls:
            web_docs = self.load_urls(urls)
            all_docs.extend(web_docs)
            print(f"  Loaded {len(web_docs)} docs from web URLs")

        print(f"  Total documents loaded: {len(all_docs)}")
        return all_docs


# Quick test — run this file directly to verify ingestion works
if __name__ == "__main__":
    ingester = DocumentIngester()

    # Drop any PDF or .md file into data/raw/ and run this
    docs = ingester.ingest(
        urls=["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
    )

    print(f"\nSample document:")
    if docs:
        print(f"  Source: {docs[0].metadata}")
        print(f"  Content preview: {docs[0].page_content[:300]}...")