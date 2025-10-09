
import json
import pandas as pd
from collections import defaultdict

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_rankings(data):
    extracted_data = []
    for item in data:
        task_id = item['id']
        question = item['data']['question']
        answers = {ans['id']: ans['html'] for ans in item['data']['answers']}
        
        # Assuming there are two annotations per task_id
        for annotation in item['annotations']:
            annotator_email = annotation['completed_by']['email']
            rank_list = annotation['result'][0]['value']['ranker']['rank']
            
            extracted_data.append({
                'task_id': task_id,
                'question': question,
                'annotator': annotator_email,
                'rankings': rank_list,
                'answers': answers
            })
    return extracted_data

def calculate_agreement(rankings_per_task):
    agreements = []
    for task_id, ranks in rankings_per_task.items():
        if len(ranks) == 2:
            rank1 = ranks[0]
            rank2 = ranks[1]
            
            # Calculate Kendall's Tau or Spearman's Rho for agreement
            # For simplicity, let's just check exact match for now
            exact_match = (rank1 == rank2)
            
            # Calculate agreement based on top-k matches
            top_k_matches = {}
            for k in [1, 3, 5]: # Check agreement for top 1, 3, and 5 results
                matches = 0
                for i in range(min(k, len(rank1), len(rank2))):
                    if rank1[i] == rank2[i]:
                        matches += 1
                top_k_matches[f'top_{k}_agreement'] = matches / k if k > 0 else 0
            
            agreements.append({
                'task_id': task_id,
                'annotator1_rank': rank1,
                'annotator2_rank': rank2,
                'exact_match': exact_match,
                **top_k_matches
            })
    return agreements

def analyze_rankings(extracted_data):
    model_scores = defaultdict(lambda: defaultdict(int))
    model_positions = defaultdict(lambda: defaultdict(list))
    
    for entry in extracted_data:
        rankings = entry['rankings']
        for i, model_id in enumerate(rankings):
            # Assign scores based on rank (e.g., 9 for 1st, 1 for 9th)
            score = len(rankings) - i
            model_scores[model_id]['total_score'] += score
            model_scores[model_id]['count'] += 1
            model_positions[model_id][i+1].append(1) # Store occurrence at each rank
            
    # Calculate average score and average rank
    model_performance = {}
    for model_id, data in model_scores.items():
        total_score = data['total_score']
        count = data['count']
        avg_score = total_score / count if count > 0 else 0
        
        # Calculate average rank
        total_rank_sum = 0
        total_rank_count = 0
        for rank, occurrences in model_positions[model_id].items():
            total_rank_sum += rank * len(occurrences)
            total_rank_count += len(occurrences)
        avg_rank = total_rank_sum / total_rank_count if total_rank_count > 0 else 0
        
        model_performance[model_id] = {
            'total_score': total_score,
            'count': count,
            'average_score': avg_score,
            'average_rank': avg_rank
        }
        
    return model_performance

# Main execution
raw_json_data = load_json_data('/home/ubuntu/upload/NLP_HW2_Phase5_labels_raw_json.json')

extracted_rankings = extract_rankings(raw_json_data)

# Group rankings by task_id to calculate inter-annotator agreement
rankings_by_task = defaultdict(list)
for entry in extracted_rankings:
    rankings_by_task[entry['task_id']].append(entry['rankings'])

agreement_results = calculate_agreement(rankings_by_task)

model_performance = analyze_rankings(extracted_rankings)

# Output results to files
with open('agreement_results.json', 'w', encoding='utf-8') as f:
    json.dump(agreement_results, f, indent=4)

with open('model_performance.json', 'w', encoding='utf-8') as f:
    json.dump(model_performance, f, indent=4)

print("Analysis complete. Results saved to agreement_results.json and model_performance.json")


