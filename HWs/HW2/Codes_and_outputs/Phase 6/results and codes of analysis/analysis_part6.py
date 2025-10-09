import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict
import re

# No specific Persian font needed for English labels

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_rankings_and_questions(data):
    extracted_data = []
    question_types = defaultdict(list)
    for item in data:
        task_id = item['id']
        question = item['data']['question']
        answers = {ans['id']: ans['html'] for ans in item['data']['answers']}
        
        # Simple categorization of questions based on keywords
        if "در چه شهری" in question or "اهل کدام شهر" in question or "در کجا" in question:
            q_type = "Location (City/Province)"
        elif "در چه دهه‌ای" in question or "در تاریخ" in question or "در سال" in question:
            q_type = "Time (Date/Year)"
        elif "چه کسی" in question or "کدام فرد" in question:
            q_type = "Person"
        elif "چه رشته‌ای" in question or "چه فعالیتی" in question:
            q_type = "Activity/Field"
        else:
            q_type = "Other"
        question_types[q_type].append(task_id)

        for annotation in item['annotations']:
            annotator_email = annotation['completed_by']['email']
            rank_list = annotation['result'][0]['value']['ranker']['rank']
            
            extracted_data.append({
                'task_id': task_id,
                'question': question,
                'question_type': q_type,
                'annotator': annotator_email,
                'rankings': rank_list,
                'answers': answers
            })
    return extracted_data, question_types

def analyze_rankings_by_question_type(extracted_data, question_types):
    model_performance_by_type = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for entry in extracted_data:
        q_type = entry['question_type']
        rankings = entry['rankings']
        for i, model_id in enumerate(rankings):
            score = len(rankings) - i # 9 for 1st, 1 for 9th
            model_performance_by_type[q_type][model_id]['total_score'] += score
            model_performance_by_type[q_type][model_id]['count'] += 1
            model_performance_by_type[q_type][model_id]['total_rank_sum'] += (i + 1)

    results = {}
    for q_type, models_data in model_performance_by_type.items():
        results[q_type] = {}
        for model_id, data in models_data.items():
            avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
            avg_rank = data['total_rank_sum'] / data['count'] if data['count'] > 0 else 0
            results[q_type][model_id] = {
                'average_score': avg_score,
                'average_rank': avg_rank
            }
    return results

def plot_performance_by_question_type(performance_by_type):
    for q_type, models_data in performance_by_type.items():
        df = pd.DataFrame.from_dict(models_data, orient='index')
        df = df.sort_values(by='average_score', ascending=False)

        fig, ax1 = plt.subplots(figsize=(12, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Average Score', color=color)
        ax1.bar(df.index, df['average_score'], color=color, alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticklabels(df.index, rotation=45, ha='right')

        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Average Rank', color=color)  
        ax2.plot(df.index, df['average_rank'], color=color, marker='o', linestyle='-', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.invert_yaxis() 

        fig.tight_layout()
        plt.title(f'Model Performance by Question Type: {q_type}')
        
        # Sanitize filename
        sanitized_q_type = re.sub(r'[^\w_.-]', '', q_type.replace(' ', '_'))
        plt.savefig(f'model_performance_by_type_{sanitized_q_type}.png')
        plt.close(fig)

    print("Charts generated for each question type.")

# Main execution
raw_json_data = load_json_data('/home/ubuntu/upload/NLP_HW2_Phase5_labels_raw_json.json')

extracted_data, question_types = extract_rankings_and_questions(raw_json_data)

performance_by_type = analyze_rankings_by_question_type(extracted_data, question_types)

with open('model_performance_by_question_type.json', 'w', encoding='utf-8') as f:
    json.dump(performance_by_type, f, indent=4)

plot_performance_by_question_type(performance_by_type)

print("Detailed analysis for Part 6 complete. Results saved to model_performance_by_question_type.json and corresponding charts.")



