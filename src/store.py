from __future__ import annotations

import uuid
from typing import Any, Callable

import chromadb

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Uses ChromaDB for storage.
    The embedding_fn parameter allows injection of embeddings.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path="./chroma_db")
        
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
            
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, docs: list[Document]) -> None:
        """Embed each document's content and store it."""
        if not docs:
            return
            
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for doc in docs:
            ids.append(str(uuid.uuid4()))
            documents.append(doc.content)
            
            emb = self._embedding_fn(doc.content)
            embeddings.append(emb)
            
            meta = {**doc.metadata, "doc_id": doc.id}
            metadatas.append(meta)
            
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents to query."""
        query_embedding = self._embedding_fn(query)
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        ret = []
        if results and results["ids"]:
            num_results = len(results["ids"][0])
            for i in range(num_results):
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                score = 1.0 - distance
                
                ret.append({
                    "content": content,
                    "metadata": metadata,
                    "score": score
                })
            
        return ret

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return self._collection.count()

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """Search with optional metadata pre-filtering."""
        query_embedding = self._embedding_fn(query)
        
        where = metadata_filter if metadata_filter else None
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        ret = []
        if results and results["ids"]:
            num_results = len(results["ids"][0])
            for i in range(num_results):
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                score = 1.0 - distance
                
                ret.append({
                    "content": content,
                    "metadata": metadata,
                    "score": score
                })
            
        return ret

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks belonging to a document."""
        initial_count = self.get_collection_size()
        
        self._collection.delete(
            where={"doc_id": doc_id}
        )
        
        return self.get_collection_size() < initial_count
