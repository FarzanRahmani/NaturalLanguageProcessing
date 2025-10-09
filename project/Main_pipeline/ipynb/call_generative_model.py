import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

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
