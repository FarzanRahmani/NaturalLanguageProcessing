from config import Config
import pickle
from pathlib import Path

from image_pipeline.image_embedder import FaceRecognizer
from image_pipeline.image_vector_database import ImageVectorDatabase


def init_image_pipeline():
    print("Initializing Face Recognizer...")
    recognizer = FaceRecognizer()

    Config.CACHE_PATH.mkdir(exist_ok=True)

    paths = []
    embeddings = []

    if Config.IMG_EMBS_CACHE_PATH.exists():
        print(f"Loading cached embeddings from {Config.IMG_EMBS_CACHE_PATH}...")
        with open(Config.IMG_EMBS_CACHE_PATH, 'rb') as f:
            data = pickle.load(f)
            paths, embeddings = data["paths"], data["embeddings"]

        cached_paths_set = set(paths)
        all_image_files = [str(p) for p in Config.GT_IMG_DATASET_PATH.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        new_image_paths = [Path(p) for p in all_image_files if p not in cached_paths_set]

        if new_image_paths:
            print(f"Found {len(new_image_paths)} new images to process.")
            new_paths, new_embeddings = recognizer.get_embeddings_for_paths(new_image_paths)

            paths.extend(new_paths)
            embeddings.extend(new_embeddings)

            print(f"Updating cache file at {Config.IMG_EMBS_CACHE_PATH}...")
            with open(Config.IMG_EMBS_CACHE_PATH, 'wb') as f:
                pickle.dump({"paths": paths, "embeddings": embeddings}, f)
            print("Cache updated successfully.")
        else:
            print("Cache is up to date. No new images found.")

    else:
        print(f"Cache not found. Building embeddings from scratch for all images in {Config.GT_IMG_DATASET_PATH}...")
        paths, embeddings = recognizer.create_database_embeddings(Config.GT_IMG_DATASET_PATH)

        if paths:
            with open(Config.IMG_EMBS_CACHE_PATH, 'wb') as f:
                pickle.dump({"paths": paths, "embeddings": embeddings}, f)
            print(f"Embeddings cached to {Config.IMG_EMBS_CACHE_PATH}")

    if not embeddings:
        print("No embeddings found in the database. Exiting.")
        exit()

    print("Initializing Vector Database...")
    vector_db = ImageVectorDatabase(collection_name="insightface_faces")
    vector_db.add_batch(paths, embeddings)
    print(f"Vector database ready with {vector_db.collection.count()} entries.")

    return recognizer, vector_db