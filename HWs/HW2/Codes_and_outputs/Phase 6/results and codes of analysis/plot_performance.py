
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_model_performance(model_performance_data):
    df = pd.DataFrame.from_dict(model_performance_data, orient='index')
    df = df.sort_values(by='average_score', ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Bar chart for average score
    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Average Score', color=color)
    ax1.bar(df.index, df['average_score'], color=color, alpha=0.6, label='Average Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(df.index, rotation=45, ha='right')

    # Line plot for average rank
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Average Rank', color=color)  
    ax2.plot(df.index, df['average_rank'], color=color, marker='o', linestyle='-', linewidth=2, label='Average Rank')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.invert_yaxis() # Lower rank is better

    fig.tight_layout()
    plt.title('Model Performance: Average Score and Average Rank')
    fig.savefig('model_performance_chart.png')
    plt.close(fig)

def plot_agreement_metrics(agreement_data):
    df = pd.DataFrame(agreement_data)

    # Calculate overall averages
    avg_top1 = df['top_1_agreement'].mean()
    avg_top3 = df['top_3_agreement'].mean()
    avg_top5 = df['top_5_agreement'].mean()

    metrics = ['Top-1 Agreement', 'Top-3 Agreement', 'Top-5 Agreement']
    values = [avg_top1, avg_top3, avg_top5]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_ylabel('Average Agreement')
    ax.set_title('Average Inter-Annotator Agreement')
    ax.set_ylim(0, 1.1) # Set y-axis limit from 0 to 1.1 for better visualization
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')
    fig.savefig('agreement_metrics_chart.png')
    plt.close(fig)

# Main execution
model_performance_data = load_json_data("model_performance.json")
plot_model_performance(model_performance_data)

agreement_data = load_json_data("agreement_results.json")
plot_agreement_metrics(agreement_data)

print("Charts generated: model_performance_chart.png and agreement_metrics_chart.png")



