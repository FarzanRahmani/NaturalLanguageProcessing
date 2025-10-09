import json

def extract_short_questions_from_file(json_path, max_length=160):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result_questions = []

    for item in data:
        text = item.get("textual_content_response", "")
        if len(text.strip()) <= max_length:
            qna = item.get("question_and_answer_response", {}).get("questions_and_answers", [])
            for qa in qna:
                question_text = qa.get("question", "").strip()
                if question_text:
                    result_questions.append(question_text)

    return result_questions

if __name__ == "__main__":
    json_file = "data.json"
    questions = extract_short_questions_from_file(json_file)
    
    print("سوالاتی که متن توصیفی‌شان <= 160 کاراکتر است:")
    for q in questions:
        print("•", q)