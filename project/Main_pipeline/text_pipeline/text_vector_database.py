import chromadb
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from tqdm import tqdm


class TextVectorDatabase:
    def __init__(self, collection_name: str):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )


    def add_batch(self, ids: List[str], embeddings: List[list], metadatas: List[dict], batch_size: int = 1000):
        if not embeddings or not ids:
            print("No embeddings or IDs provided.")
            return

        num_items = len(ids)
        for i in tqdm(range(0, num_items, batch_size), desc="Adding to Vector DB"):
            end_index = min(i + batch_size, num_items)

            self.collection.add(
                ids=ids[i:end_index],
                embeddings=embeddings[i:end_index],
                metadatas=metadatas[i:end_index]
            )

        print(f"Successfully added {num_items} total items to the collection '{self.collection.name}'.")


    def search(self, query_embedding: np.ndarray, k: int = 1) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )