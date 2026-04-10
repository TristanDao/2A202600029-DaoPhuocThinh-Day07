from __future__ import annotations

import uuid
from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use Qdrant if available; falls back to an in-memory list store.
    The embedding_fn parameter allows injection of embeddings.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_qdrant = False
        self._store: list[dict[str, Any]] = []
        self._client = None
        self._next_index = 0

        try:
            import os
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams

            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")

            if qdrant_url:
                # Connect to Qdrant Cloud or remote server
                self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                # Fallback to local in-memory
                self._client = QdrantClient(location=":memory:")
            
            # Check if collection exists to avoid deleting cloud data on every run
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self._collection_name not in collection_names:
                dummy_vec = self._embedding_fn("test")
                dim = len(dummy_vec)
                self._client.recreate_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            
            self._use_qdrant = True
        except Exception:
            self._use_qdrant = False
            self._client = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record for one document."""
        # Ensure metadata contains doc_id for search/delete consistency
        metadata = {**doc.metadata, "doc_id": doc.id}
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over provided records."""
        query_vec = self._embedding_fn(query)
        scored_records = []
        for rec in records:
            score = compute_similarity(query_vec, rec["embedding"])
            scored_records.append({**rec, "score": score})
        
        # Sort by score descending
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """Embed each document's content and store it."""
        if self._use_qdrant:
            from qdrant_client.http.models import PointStruct
            
            points = []
            for doc in docs:
                embedding = self._embedding_fn(doc.content)
                # Store doc.id in metadata as doc_id for filtering
                metadata = {**doc.metadata, "doc_id": doc.id}
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()), # Point ID must be UUID or int
                        vector=embedding,
                        payload={"content": doc.content, "metadata": metadata}
                    )
                )
            self._client.upsert(
                collection_name=self._collection_name,
                points=points
            )
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents to query."""
        if self._use_qdrant:
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=self._embedding_fn(query),
                limit=top_k
            )
            return [
                {
                    "content": res.payload["content"],
                    "metadata": res.payload["metadata"],
                    "score": res.score
                }
                for res in results
            ]
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_qdrant:
            return self._client.get_collection(self._collection_name).points_count
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """Search with optional metadata pre-filtering."""
        if self._use_qdrant:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue
            
            q_filter = None
            if metadata_filter:
                must = []
                for key, value in metadata_filter.items():
                    # Qdrant payload path
                    must.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))
                q_filter = Filter(must=must)
            
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=self._embedding_fn(query),
                query_filter=q_filter,
                limit=top_k
            )
            return [
                {
                    "content": res.payload["content"],
                    "metadata": res.payload["metadata"],
                    "score": res.score
                }
                for res in results
            ]
        else:
            filtered_records = self._store
            if metadata_filter:
                filtered_records = [
                    rec for rec in self._store
                    if all(rec["metadata"].get(k) == v for k, v in metadata_filter.items())
                ]
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks belonging to a document."""
        if self._use_qdrant:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue
            
            # Count points before to see if any were deleted
            # Qdrant delete doesn't return count directly in some versions, but we can filter
            initial_count = self.get_collection_size()
            
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_id))
                    ]
                )
            )
            return self.get_collection_size() < initial_count
        else:
            new_store = [
                rec for rec in self._store
                if rec["metadata"].get("doc_id") != doc_id
            ]
            deleted = len(new_store) < len(self._store)
            self._store = new_store
            return deleted
