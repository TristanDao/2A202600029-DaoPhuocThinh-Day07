from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve top-k relevant chunks from the store
        chunks = self.store.search(question, top_k=top_k)
        
        # 2. Build a prompt with the chunks as context
        context_text = "\n\n".join([c["content"] for c in chunks])
        prompt = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{context_text}\n"
            f"---------------------\n"
            f"Given the context information and not prior knowledge, "
            f"answer the question: {question}\n"
        )
        
        # 3. Call the LLM to generate an answer
        return self.llm_fn(prompt)
