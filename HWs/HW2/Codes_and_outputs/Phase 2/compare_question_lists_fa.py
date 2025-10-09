from sentence_similarity_fa import sentence_similarity

def find_low_similarity_questions(list_a, list_b, threshold=20.0):
    result = []

    for idx_a, q1 in enumerate(list_a):
        similarities = []
        for q2 in list_b:
            score, _ = sentence_similarity(q1, q2)
            similarities.append(score)
        average_score = sum(similarities) / len(similarities)
        if average_score < threshold:
            result.append((q1, average_score))

    # Sort by similarity score (ascending)
    result.sort(key=lambda x: x[1])
    return result

if __name__ == "__main__":
    list_a = [
        "تاریخ تولد حافظ شیرازی چه زمانی است؟",
        "چه کسی نخستین رئیس جمهور ایران بود؟",
        "چگونه می‌توان یک سیب زمینی را پخت؟",
        "چه کسی گل اول را در بازی زد؟"
    ]

    list_b = [
        "چه کسی بنیان‌گذار جمهوری اسلامی ایران است؟",
        "رئیس جمهور کنونی ایران کیست؟",
        "نخستین انتخابات ریاست جمهوری چه زمانی برگزار شد؟"
    ]

    filtered_questions = find_low_similarity_questions(list_a, list_b)
    for q, score in filtered_questions:
        print(f"سؤال: {q}")
        print(f"میانگین شباهت با لیست دوم: {round(score, 2)}%")
        print("---")