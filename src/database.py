import os
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

MODEL_DIR = os.path.join("../models", "embedding_models")
DB_DIR = os.path.join("../db")


class VectorDatabase:
    def __init__(self, doc_name):
        self.doc_name = doc_name

        self.doc_path = os.path.join(DB_DIR, doc_name)

        self.vectors_path = os.path.join(self.doc_path, "vectors.npy")
        self.metadata_path = os.path.join(self.doc_path, "metadata.json")

        if not os.path.exists(self.vectors_path):
            raise FileNotFoundError(f"Vectors not found at {self.vectors_path}")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")

        # Load embeddings
        self.vectors = np.load(self.vectors_path).astype("float32")

        # Load metadata
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load embedding model
        self.model = SentenceTransformer(MODEL_DIR, trust_remote_code=True)

        # Create FAISS index
        self.dimension = self.vectors.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(self.vectors)

        print(f"Loaded {len(self.vectors)} vectors into FAISS.")


    def embed_query(self, query):
        query_embedding = self.model.encode(
            [f"search_query: {query}"],
            convert_to_numpy=True
        )

        return query_embedding.astype("float32")


    def search(self, query, top_k=5):
        query_vector = self.embed_query(query)

        distances, indices = self.index.search(query_vector, top_k)

        results = []

        for i, idx in enumerate(indices[0]):
            if idx >= len(self.metadata):
                continue

            result = {
                "rank": i + 1,
                "score": float(distances[0][i]),
                "chunk": self.metadata[idx]["chunk"],
                "type": self.metadata[idx]["type"]
            }

            results.append(result)

        return results


    def pretty_print(self, results):
        print("\nSearch Results:\n")

        for result in results:
            print("=" * 60)
            print(f"Rank : {result['rank']}")
            print(f"Type : {result['type']}")
            print(f"Score: {result['score']:.4f}")
            print("-" * 60)
            print(result["chunk"])
            print("=" * 60)
            print()