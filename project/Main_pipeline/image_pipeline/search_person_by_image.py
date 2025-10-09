from pathlib import Path

from utils import get_info_from_dataset
from utils import extract_image_fields
from image_pipeline.image_embedder import FaceRecognizer
from image_pipeline.image_vector_database import ImageVectorDatabase


def search_person_by_image(
        query_image_path: Path,
        n_results: int,
        recognizer: FaceRecognizer,
        vector_db: ImageVectorDatabase,
        json_files
    ) -> list | None:
    query_embedding = recognizer.get_embedding(query_image_path)
    if query_embedding is None:
        print(f"Could not get embedding for {query_image_path}. No face found or image is invalid.")
        return None

    results = vector_db.search(query_embedding, k=n_results)

    if not results or not results['ids'][0]:
        print("No matches found in the database for this person.")
        return None

    best_matches = results['metadatas'][0]
    distances = results['distances'][0]

    search_results = []
    for idx, (metadata, distance) in enumerate(zip(best_matches, distances)):
        json_id = "json_" + metadata["path"].split("/")[-3].split("_")[-1]
        iteration = int(metadata["path"].split("/")[-2])
        person_info = get_info_from_dataset(json_id, iteration, json_files)

        item = {
            "nth_match": idx + 1,
            "json_id": json_id,
            "iteration": iteration,
            "person_name": person_info["json_pure"]["name"],
            "distance": distance,
            # "json": person_info["json_refined"],
            "textual_content": person_info["textual_content_response"],
            # "question_and_answer": person_info["question_and_answer_response"],
            "image_data": extract_image_fields(person_info["image_data"]),
            "match_image_path": metadata["path"],
        }
        search_results.append(item)
    return search_results