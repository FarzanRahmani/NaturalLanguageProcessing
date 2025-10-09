from text_pipeline.init_text_pipeline import init_text_pipeline
from text_pipeline.person_name_extractor import PersonNameExtractor
from image_pipeline.init_image_pipeline import init_image_pipeline
from utils import merge_data_sources

def setup():
    people_list_total, json_files = merge_data_sources()
    text_embedder, text_vector_db = init_text_pipeline(people_list_total)
    face_recognizer, image_vector_db = init_image_pipeline()
    person_name_extractor = PersonNameExtractor()
    return (
        text_embedder, 
        text_vector_db,
        face_recognizer,
        image_vector_db,
        person_name_extractor,
        people_list_total, 
        json_files
    )


if __name__ == "__main__":
    setup()