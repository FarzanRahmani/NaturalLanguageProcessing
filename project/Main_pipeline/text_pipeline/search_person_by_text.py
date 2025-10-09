from text_pipeline.person_name_extractor import PersonNameExtractor
from text_pipeline.text_embedder import TextEmbedder
from text_pipeline.text_vector_database import TextVectorDatabase
from utils import get_info_from_dataset
from utils import extract_image_fields

def search_person_by_text(
        text_query: str,
        n_results: int,
        person_name_extractor: PersonNameExtractor,
        embedder: TextEmbedder,
        vector_db: TextVectorDatabase,
        json_files
    ) -> list | None:
    person_names = person_name_extractor.extract(text_query)
    if not person_names:
        print("No person name found!")
        return None
    query_embedding = embedder.get_embedding(person_names[0])
    results = vector_db.search(query_embedding, k=n_results)

    best_matches = results['metadatas'][0]
    distances = results['distances'][0]

    search_results = []
    for idx, (metadata, distance) in enumerate(zip(best_matches, distances)):
        print("metadata", metadata)
        person_info = get_info_from_dataset(metadata["json_id"],
                                            metadata["iteration"], 
                                            json_files)
        item = {
            "nth_match": idx + 1,
            "json_id": metadata["json_id"],
            "iteration": metadata["iteration"],
            "person_name": person_info["json_pure"]["name"],
            "distance": distance,
            # "json": person_info["json_refined"],
            "textual_content": person_info["textual_content_response"],
            # "question_and_answer": person_info["question_and_answer_response"],
            "image_data": extract_image_fields(person_info["image_data"]),
        }
        search_results.append(item)
    return search_results