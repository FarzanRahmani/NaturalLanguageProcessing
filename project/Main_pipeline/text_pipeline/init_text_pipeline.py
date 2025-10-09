from typing import Any, Dict, List, Tuple, Union
import pickle

from text_pipeline.text_embedder import TextEmbedder
from text_pipeline.text_vector_database import TextVectorDatabase
from config import Config



def init_text_pipeline(source_data: List[Dict[str, Any]]) -> Tuple[TextEmbedder, TextVectorDatabase]:
    print("Initializing Text Embedder...")
    embedder = TextEmbedder(model_name=Config.TEXT_EMB_MODEL, model_provider=Config.TEXT_EMB_PROV)

    ids, metadatas, embeddings = [], [], []

    if Config.TXT_EMBS_CACHE_PATH.exists():
        print(f"Loading cached embeddings from {Config.TXT_EMBS_CACHE_PATH}...")
        with open(Config.TXT_EMBS_CACHE_PATH, 'rb') as f:
            cached_data = pickle.load(f)
            ids, metadatas, embeddings = cached_data["ids"], cached_data["metadatas"], cached_data["embeddings"]

        cached_json_ids = {meta['json_id'] for meta in metadatas}
        new_items_to_process = [item for item in source_data if item['json_id'] not in cached_json_ids]

        if new_items_to_process:
            print(f"Found {len(new_items_to_process)} new items to embed.")
            new_ids, new_metadatas, new_embeddings = embedder.create_database_embeddings(
                new_items_to_process, start_index=len(ids)
            )

            ids.extend(new_ids)
            metadatas.extend(new_metadatas)
            embeddings.extend(new_embeddings)

            with open(Config.TXT_EMBS_CACHE_PATH, 'wb') as f:
                pickle.dump({"ids": ids, "metadatas": metadatas, "embeddings": embeddings}, f)
            print(f"Cache updated with new embeddings at {Config.TXT_EMBS_CACHE_PATH}")
        else:
            print("Cache is already up to date.")

    else:
        print(f"Cache not found. Building embeddings from scratch for {len(source_data)} items...")
        ids, metadatas, embeddings = embedder.create_database_embeddings(source_data)

        with open(Config.TXT_EMBS_CACHE_PATH, 'wb') as f:
            pickle.dump({"ids": ids, "metadatas": metadatas, "embeddings": embeddings}, f)
        print(f"Embeddings cached to {Config.TXT_EMBS_CACHE_PATH}")

    if not embeddings:
        print("No embeddings were generated or loaded. Exiting.")
        exit()

    print("Initializing Vector Database for text...")
    vector_db = TextVectorDatabase(collection_name=f"{Config.TEXT_EMB_PROV}_person_names")
    vector_db.add_batch(ids, embeddings, metadatas)

    print(f"Text vector database ready with {vector_db.collection.count()} entries.")
    return embedder, vector_db