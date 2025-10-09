import json
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Union

from config import Config
from PIL import Image

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()



def show_image_with_title(image_path: str, title_text: str = ""):
    try:
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title_text)
        plt.show()
    except FileNotFoundError:
        print(f"Error: Could not display file at '{image_path}'")
    except Exception as e:
        print(f"An error occurred while displaying image: {e}")



def read_json_file(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        return None
    


def create_people_list(json_data: dict, json_id):
    people_list = []
    for idx, item in enumerate(json_data):
        item_summary = {"iteration": item["iteration"], "json_id": json_id}
        try:
            if type(item["json_pure"]["name"]) is list:
                person_name = " ".join(item["json_pure"]["name"]).replace('\u200c', ' ')
            else:
                person_name = item["json_pure"]["name"].replace('\u200c', ' ')
            item_summary["person_name"] = person_name
        except:
            continue
            item_summary["person_name"] = "unknown"
        people_list.append(item_summary)
    return people_list




def dict_pretty_print(data_to_print: list | dict):
    formatted_json = json.dumps(
        data_to_print,
        indent=4,
        ensure_ascii=False
    )
    print(formatted_json)





def get_info_from_dataset(json_id: str, iteration: int, json_files: list) -> dict:

    json_1, json_3, json_4, json_7 = json_files

    if json_id == "json_1":
        data_source = json_1
    elif json_id == "json_3":
        data_source = json_3
    elif json_id == "json_4":
        data_source = json_4
    elif json_id == "json_7":
        data_source = json_7

    if data_source[iteration]["iteration"] == iteration:
        return data_source[iteration]
    else:
        for item in data_source:
            if item["iteration"] == iteration:
                return item
            

        

def extract_image_fields(data: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    extracted_data = [
        {
            "url": image.get("url"),
            "description": image.get("description"),
            "idx": image.get("idx"),
        }
        for image in data.get("image_urls", [])
    ]
    return extracted_data



def merge_data_sources():
    json_1 = read_json_file(Config.GT_IMG_JSON_PATH_1)
    json_3 = read_json_file(Config.GT_IMG_JSON_PATH_3)
    json_4 = read_json_file(Config.GT_IMG_JSON_PATH_4)
    json_7 = read_json_file(Config.GT_IMG_JSON_PATH_7)

    json_1_people_list = create_people_list(json_1, "json_1")
    json_3_people_list = create_people_list(json_3, "json_3")
    json_4_people_list = create_people_list(json_4, "json_4")
    json_7_people_list = create_people_list(json_7, "json_7")


    people_list_total = [*json_1_people_list,
                        *json_3_people_list,
                        *json_4_people_list,
                        *json_7_people_list]
    return people_list_total, [json_1, json_3, json_4, json_7]






# from google.colab import userdata
# os.environ["GEMINI_API_KEY"] = userdata.get('Gemeni_API_KEY')

def _infer_mime_type_from_path(path: str) -> str:
    """
    Helper to guess mime_type from file extension.
    Extend this if needed (png, jpg, jpeg, gif, etc).
    """
    lower = path.lower()
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    elif lower.endswith(".png"):
        return "image/png"
    elif lower.endswith(".gif"):
        return "image/gif"
    else:
        # default fallback
        return "application/octet-stream"


SYSTEM_PROMPT_IRANIAN_FIGURES = """
شما یک مدل هوش مصنوعی هستید که تخصص‌تان در پاسخ‌دهی درباره مشاهیر ایرانی است.

اگر در زمینه اطلاعات مرتبط برای پاسخ به سوال وجود دارد از آن استفاده کن و اگر مرتبط نیست یا وجود ندارد از دانش عمومی خودت استفاده کن.

سوال 4 گزینه است لطفا فقط یکی از گزینه ها را انتخاب کن و فقط در یک کلمه یک عدد از 1 تا 4 انتخاب کن.
"""

def call_generative_model(
    input_text: str | None,
    input_image_path: str | None,
    retrieved_texts: list[str] | None = None,
    retrieved_images_paths: list[str] | None = None,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.25,
    top_p: float = 0.5,
    max_output_tokens: int = 256,
    thinking_budget: int = 0
) -> str:
    """
    Calls a Gemini generative model with optional text & image context,
    incorporating retrieved texts/images, and optional main image + text.

    Either or both of input_text / input_image_path can be None,
    similarly retrieved_texts or retrieved_images_paths can be None.

    Args:
        model_name: name of the Gemini model, e.g. "gemini-2.5-flash"
        input_text: main user prompt / query (may be None if prompt only image)
        input_image_path: path to image file for main prompt (may be None)
        retrieved_texts: list of text strings retrieved by retriever (or None)
        retrieved_images_paths: list of image file paths retrieved as context (or None)
        temperature, top_p, max_output_tokens, thinking_budget: config params

    Returns:
        Generated text from the model.

    Raises:
        ValueError if both input_text and input_image_path are None (no prompt).
    """

    # Basic checks
    if input_text is None and input_image_path is None:
        raise ValueError("At least one of input_text or input_image_path must be provided for prompt.")

    client = genai.Client()

    contents: list[types.Part | str] = []

    # 1. Retrieved text context
    if retrieved_texts:
        # Format nicely, maybe separate them
        # text_context = "Context from retrieved texts:\n" + "\n---\n".join(retrieved_texts)
        text_context = "زمینه مرتبط از متون بازیابی‌شده:\n" + "\n---\n".join(retrieved_texts)
        contents.append(text_context)

    # 2. Retrieved images as context
    if retrieved_images_paths:
        for img_path in retrieved_images_paths:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            part = types.Part.from_bytes(
                data=img_bytes,
                mime_type=_infer_mime_type_from_path(img_path)
            )
            contents.append(part)

    # 3. Main input image
    if input_image_path:
        with open(input_image_path, "rb") as f:
            main_img_bytes = f.read()
        main_img_part = types.Part.from_bytes(
            data=main_img_bytes,
            mime_type=_infer_mime_type_from_path(input_image_path)
        )
        contents.append(main_img_part)

    # 4. Main input text
    if input_text:
        contents.append(input_text)

    # 5. Generation config
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        # system_instruction=SYSTEM_PROMPT_IRANIAN_FIGURES
    )

    # 6. Call Gemini
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config
    )

    return response.text



