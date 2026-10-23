"""
vector_store.py
----------------
Thin wrapper around a FAISS index that also persists the text + metadata
for each vector (FAISS itself only stores numbers, not the chunk text
or which document/file a vector came from).

Design notes:
- Uses IndexIDMap over IndexFlatIP (inner product) so each vector gets a
  stable integer ID we control, which makes per-document deletion
  possible (plain IndexFlatIP has no delete/remove support).
- Vectors are L2-normalized before insertion, which makes inner product
  equivalent to cosine similarity — the standard setup for semantic
  search with FAISS.
- Metadata (chunk text, filename, doc_id) lives in a parallel JSON file
  keyed by the same integer IDs, so a search hit can be mapped straight
  back to readable text.
"""
import json
import os
import uuid
import numpy as np
import faiss

import config
from core.embeddings import embedding_dim


class VectorStore:
    def __init__(self):
        self.dim = embedding_dim()
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        self.metadata = {}  # str(id) -> {"doc_id", "filename", "chunk_index", "text"}
        self._next_id = 0
        self._load()

    # -- persistence --------------------------------------------------
    def _load(self):
        if os.path.exists(config.INDEX_PATH) and os.path.exists(config.METADATA_PATH):
            self.index = faiss.read_index(config.INDEX_PATH)
            with open(config.METADATA_PATH, "r") as f:
                self.metadata = json.load(f)
            if self.metadata:
                self._next_id = max(int(k) for k in self.metadata.keys()) + 1

    def _save(self):
        faiss.write_index(self.index, config.INDEX_PATH)
        with open(config.METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)

    # -- writes ---------------------------------------------------------
    def add_document(self, filename: str, chunks: list, vectors: np.ndarray):
        """
        Adds all chunks of one document to the index.
        Returns the generated doc_id so the caller can report it back
        to the client (used for later deletion / listing).
        """
        doc_id = str(uuid.uuid4())[:8]
        faiss.normalize_L2(vectors)

        ids = np.arange(self._next_id, self._next_id + len(chunks)).astype("int64")
        self.index.add_with_ids(vectors, ids)

        for i, (chunk_id, chunk_text) in enumerate(zip(ids, chunks)):
            self.metadata[str(int(chunk_id))] = {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk_text,
            }

        self._next_id += len(chunks)
        self._save()
        return doc_id

    def delete_document(self, doc_id: str) -> int:
        """Removes every chunk belonging to doc_id. Returns count removed."""
        ids_to_remove = [
            int(k) for k, v in self.metadata.items() if v["doc_id"] == doc_id
        ]
        if not ids_to_remove:
            return 0

        self.index.remove_ids(np.array(ids_to_remove, dtype="int64"))
        for k in ids_to_remove:
            del self.metadata[str(k)]
        self._save()
        return len(ids_to_remove)

    # -- reads ----------------------------------------------------------
    def search(self, query_vector: np.ndarray, top_k: int):
        if self.index.ntotal == 0:
            return []

        faiss.normalize_L2(query_vector)
        scores, ids = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            meta = self.metadata.get(str(int(idx)))
            if meta:
                results.append({
                    "score": float(score),
                    "filename": meta["filename"],
                    "doc_id": meta["doc_id"],
                    "chunk_index": meta["chunk_index"],
                    "text": meta["text"],
                })
        return results

    def list_documents(self):
        docs = {}
        for meta in self.metadata.values():
            doc_id = meta["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = {"doc_id": doc_id, "filename": meta["filename"], "chunks": 0}
            docs[doc_id]["chunks"] += 1
        return list(docs.values())


# Module-level singleton so the Flask app shares one in-memory index
# across requests instead of reloading it from disk every call.
store = VectorStore()
