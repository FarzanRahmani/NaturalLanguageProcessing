import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from insightface.app import FaceAnalysis


class FaceRecognizer:
    def __init__(self, model_pack_name: str = 'buffalo_l'):
        self.app = FaceAnalysis(name=model_pack_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, image_path: Path) -> np.ndarray | None:
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            faces = self.app.get(img)
            if faces:
                return faces[0].normed_embedding
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        return None

    def create_database_embeddings(self, GT_IMG_DATASET_PATH: Path) -> Tuple[List[str], List[list]]:
        paths = []
        embeddings = []
        image_files = [p for p in GT_IMG_DATASET_PATH.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        for img_path in tqdm(image_files, desc="Building embedding database"):
            embedding = self.get_embedding(img_path)
            if embedding is not None:
                paths.append(str(img_path))
                embeddings.append(embedding.tolist())
        return paths, embeddings

    def get_embeddings_for_paths(self, image_paths: List[Path]) -> Tuple[List[str], List[list]]:
        paths = []
        embeddings = []

        for img_path in tqdm(image_paths, desc="Computing embeddings for new images"):
            embedding = self.get_embedding(img_path)
            if embedding is not None:
                paths.append(str(img_path))
                embeddings.append(embedding.tolist())

        return paths, embeddings