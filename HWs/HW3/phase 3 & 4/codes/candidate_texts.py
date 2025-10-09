import json
import random

input_file = "0000_all_img_link_added.json"

with open(input_file, "r", encoding = "utf-8") as f:
    data = json.load(f)

sampled = random.sample(data, min(20, len(data)))

for item in sampled:
    iteration = item.get("iteration")
    text = item.get("textual_content_response", "")
    print(f"{iteration}: {text}\n")