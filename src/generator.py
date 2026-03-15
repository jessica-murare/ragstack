# src/generator.py
from typing import Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# This prompt is the heart of citation enforcement —
# the model is explicitly told to say "I don't know"
# if the context doesn't support the answer
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise assistant that answers questions strictly based on the provided context.

CONTEXT:
{context}

RULES:
- Answer ONLY using information from the context above
- If the context does not contain enough information to answer, respond with:
  "I cannot answer this based on the available documents."
- Always cite which part of the context supports your answer
- Be concise and factual

QUESTION: {question}

ANSWER:"""
)


class RAGGenerator:
    """
    Takes retrieved chunks + a question, sends them to Ollama,
    returns a grounded answer with source citations.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.0,
        prompt_template: str = None,   
    ):
        self.llm = OllamaLLM(model=model, temperature=temperature)
        # Use passed template or fall back to default
        template = prompt_template or RAG_PROMPT.template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

    def _format_context(self, chunks: list[Document]) -> str:
        """
        Format retrieved chunks into a numbered context block.
        Numbers make it easy for the model to cite sources.
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            parts.append(f"[{i}] {chunk.page_content}\n(Source: {source})")
        return "\n\n---\n\n".join(parts)

    def generate(
        self,
        question: str,
        chunks: list[Document],
        verbose: bool = False
    ) -> dict:
        """
        Main entry point. Returns a dict with:
        - answer: the grounded response
        - sources: list of source URLs used
        - num_chunks: how many chunks were used as context
        """
        if not chunks:
            return {
                "answer": "I cannot answer this — no relevant documents were retrieved.",
                "sources": [],
                "num_chunks": 0,
            }

        context = self._format_context(chunks)

        if verbose:
            print(f"\n  Context sent to LLM ({len(chunks)} chunks):")
            print(f"  {context[:300]}...")

        # Build the full prompt and call Ollama
        chain = self.prompt | self.llm
        answer = chain.invoke({
            "context": context,
            "question": question,
        })

        # Collect unique sources from chunk metadata
        sources = list({
            chunk.metadata.get("source", "unknown")
            for chunk in chunks
        })

        return {
            "answer": answer.strip(),
            "sources": sources,
            "num_chunks": len(chunks),
        }


# Quick test
if __name__ == "__main__":
    from ingestion import DocumentIngester
    from chunking import DocumentChunker
    from vector_store import VectorStore

    # Build the pipeline
    ingester = DocumentIngester()
    chunker = DocumentChunker()
    store = VectorStore()

    # Ingest + chunk + store (skips embed if already done)
    docs = ingester.ingest(
        urls=["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
    )
    chunks = chunker.chunk_documents(docs)
    store.add_chunks(chunks)

    # Init generator
    print("\n=== Generator ===")
    generator = RAGGenerator()

    # Test 1 — question the context CAN answer
    q1 = "How is RAG evaluated?"
    retrieved = store.retrieve(q1, top_k=3)
    result = generator.generate(q1, retrieved)
    print(f"\nQ: {q1}")
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")

    # Test 2 — question the context CANNOT answer (hallucination test)
    q2 = "What is the capital of France?"
    retrieved = store.retrieve(q2, top_k=3)
    result = generator.generate(q2, retrieved)
    print(f"\nQ: {q2}")
    print(f"A: {result['answer']}")