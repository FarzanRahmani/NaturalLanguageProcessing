import json
import random

with open('0000_json_5_phase_2.json', 'r', encoding='utf-8') as f: # 0000_json_1 or 2 or ... or 7 or 8_phase_2.json
    data = json.load(f)

short_entries = [
    item for item in data
    if len(item.get("textual_content_response", "")) <= 160
]

selected_entries = random.sample(short_entries, min(50, len(short_entries))) # عدد رندوم بین 10 تا 100 بستگی به تعداد داده ها

for i, entry in enumerate(selected_entries, 1):
    print(f"\n--- مورد شماره {i} ---")
    print("متن:")
    print(entry["textual_content_response"])
    print("سوال‌ها:")
    for qa in entry.get("question_and_answer_response", {}).get("questions_and_answers", []):
        print("-", qa["question"])