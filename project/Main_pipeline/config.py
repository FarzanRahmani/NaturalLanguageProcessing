from pathlib import Path

class Config:
    TEXT_EMB_MODEL, TEXT_EMB_PROV = "BAAI/bge-m3", "bge"
    # TEXT_EMB_MODEL, TEXT_EMB_PROV = "jinaai/jina-embeddings-v2-base-en", "jina"
    NER_MODEL = "HooshvareLab/roberta-fa-zwnj-base-ner"

    PROJECT_PATH         = Path("/home/amir/Downloads/Mashahir_rag_3/Main_pipeline")
    # GT_IMG_DATASET_PATH  = PROJECT_PATH / "face_recognition_ground_truth_images"
    # VAL_IMG_DATASET_PATH = PROJECT_PATH / "face_recognition_validation_images"

    CACHE_PATH = PROJECT_PATH / "cached_img_text_embeddings"

    IMG_EMBS_CACHE_PATH  = CACHE_PATH / "img_embeddings.pkl"
    TXT_EMBS_CACHE_PATH  = CACHE_PATH / f"txt_embeddings_{TEXT_EMB_PROV}.pkl"

    GT_IMG_DATASET_PATH = PROJECT_PATH / "json_files"

    GT_IMG_JSON_PATH_1   = GT_IMG_DATASET_PATH / "0000_all_img_link_added_1.json"
    GT_IMG_JSON_PATH_3   = GT_IMG_DATASET_PATH / "0000_all_img_link_added_3.json"
    GT_IMG_JSON_PATH_4   = GT_IMG_DATASET_PATH / "0000_all_img_link_added_4.json"
    GT_IMG_JSON_PATH_7   = GT_IMG_DATASET_PATH / "0000_all_img_link_added_7.json"