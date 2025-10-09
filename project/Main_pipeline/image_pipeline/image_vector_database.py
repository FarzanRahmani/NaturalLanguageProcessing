import chromadb
from typing import Any, Dict, List, Tuple, Union
import numpy as np


class ImageVectorDatabase:
    def __init__(self, collection_name: str = "face_db"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_batch(self, paths: List[str], embeddings: List[list], batch_size: int = 4000):
        if not embeddings or not paths:
            print("No embeddings or paths provided.")
            return

        num_items = len(paths)

        ids = [f"id_{i}" for i in range(num_items)]
        metadatas = [{"path": path} for path in paths]

        for i in range(0, num_items, batch_size):
            end_index = min(i + batch_size, num_items)

            print(f"Adding batch of {end_index - i} items to the vector database...")

            batch_embeddings = embeddings[i:end_index]
            batch_metadatas = metadatas[i:end_index]
            batch_ids = ids[i:end_index]

            self.collection.add(
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

        print(f"Successfully added {num_items} total items to the collection.")

    def search(self, query_embedding: np.ndarray, k: int = 1):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )