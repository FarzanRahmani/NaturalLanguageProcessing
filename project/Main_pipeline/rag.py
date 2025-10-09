from text_pipeline.search_person_by_text import search_person_by_text
from image_pipeline.search_person_by_image import search_person_by_image
from utils import call_generative_model


def RAG(query, 
        image_url,
        text_embedder, 
        text_vector_db,
        face_recognizer,
        image_vector_db,
        person_name_extractor,
        json_files
    ):

    # retrieved_texts = search_by_text(query=query, k=3)["metadatas"][0]
    text_retrieve_result = search_person_by_text(
        query,
        1,
        person_name_extractor,
        text_embedder,
        text_vector_db,
        json_files
    )

    if text_retrieve_result:
        retrieved_texts = text_retrieve_result[0]["textual_content"]
    else:
        retrieved_texts = None


    if image_url:
        image_retrieve_result = search_person_by_image(
            image_url,
            1,
            face_recognizer,
            image_vector_db,
            json_files
        )
        retrieved_images_paths = [image_retrieve_result[0]["match_image_path"]]
        if not retrieved_texts:
            retrieved_texts = image_retrieve_result[0]["textual_content"]
    else:
        retrieved_images_paths = None

    ans = call_generative_model(
        input_text=query,
        input_image_path=image_url,
        retrieved_texts=retrieved_texts,
        retrieved_images_paths=retrieved_images_paths,
        model_name="gemini-2.5-flash"
    )

    return ans


