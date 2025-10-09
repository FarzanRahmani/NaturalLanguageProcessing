import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Tuple, Union
from FlagEmbedding import BGEM3FlagModel



class TextEmbedder:
    def __init__(self, model_name: str, model_provider: str):
        self.model_name = model_name
        self.provider = model_provider.lower()
        print(f"Initializing embedder with provider: '{self.provider}' and model: '{self.model_name}'...")
        if self.provider == 'jina':
            self.model = SentenceTransformer(self.model_name)
        elif self.provider == 'bge':
            self.model = BGEM3FlagModel(self.model_name, use_fp16=True)
        else:
            raise ValueError("Unsupported model_provider. Please choose 'jina' or 'bge'.")
        print("Model loaded successfully.")


    def _encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if self.provider == 'jina':
            return self.model.encode(texts, show_progress_bar=True)
        elif self.provider == 'bge':
            return self.model.encode(texts)['dense_vecs']


    def get_embedding(self, text: str) -> np.ndarray:
        return self._encode(text)


    def create_database_embeddings(self, source_data: List[Dict[str, Any]], start_index: int = 0) -> Tuple[List[str], List[dict], List[list]]:
        if not source_data:
            return [], [], []

        person_names = [item['person_name'] for item in source_data]
        ids = [f"text_id_{i + start_index}" for i in range(len(source_data))]

        metadatas = [
            {
                "person_name": item['person_name'],
                "iteration": item['iteration'],
                "json_id": item['json_id']
            } for item in source_data
        ]

        print(f"Generating embeddings for {len(person_names)} names with {self.provider} model...")
        embeddings = self._encode(person_names)

        return ids, metadatas, embeddings.tolist()