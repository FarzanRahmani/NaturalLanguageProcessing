import json

with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

questions_only = [{"question": item["question"]} for item in data if "question" in item]

with open('questions.json', 'w', encoding='utf-8') as f:
    json.dump(questions_only, f, ensure_ascii=False, indent=4)

print("فایل جدید با نام 'questions.json' ذخیره شد.")