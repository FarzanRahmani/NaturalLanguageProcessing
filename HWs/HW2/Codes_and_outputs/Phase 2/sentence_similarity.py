import numpy as np
import nltk
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

# Initial downloads (do once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Global objects
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def lexical_overlap(s1, s2):
    t1, t2 = preprocess(s1), preprocess(s2)
    overlap = len(set(t1).intersection(set(t2))) / max(len(set(t1).union(set(t2))), 1)
    return overlap

def tfidf_similarity(s1, s2):
    vect = TfidfVectorizer().fit([s1, s2])
    vecs = vect.transform([s1, s2])
    return cosine_similarity(vecs[0], vecs[1])[0][0]

def bow_similarity(s1, s2):
    vect = CountVectorizer().fit([s1, s2])
    vecs = vect.transform([s1, s2])
    return cosine_similarity(vecs[0], vecs[1])[0][0]

def ngram_overlap(s1, s2, n=2):
    t1, t2 = preprocess(s1), preprocess(s2)
    n1 = set(ngrams(t1, n))
    n2 = set(ngrams(t2, n))
    return len(n1.intersection(n2)) / max(len(n1.union(n2)), 1)

def char_level_similarity(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

def pos_similarity(s1, s2):
    p1 = [tag for word, tag in pos_tag(word_tokenize(s1))]
    p2 = [tag for word, tag in pos_tag(word_tokenize(s2))]
    overlap = len(set(p1).intersection(set(p2))) / max(len(set(p1).union(set(p2))), 1)
    return overlap

def semantic_similarity_sbert(s1, s2):
    emb1 = sbert_model.encode(s1, convert_to_tensor=True)
    emb2 = sbert_model.encode(s2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)[0][0])

def sentence_similarity(s1, s2):
    features = {
        'lexical_overlap': lexical_overlap(s1, s2),
        'tfidf': tfidf_similarity(s1, s2),
        'bow': bow_similarity(s1, s2),
        'ngram': ngram_overlap(s1, s2),
        'char_level': char_level_similarity(s1, s2),
        'pos': pos_similarity(s1, s2),
        'semantic_sbert': semantic_similarity_sbert(s1, s2)
    }

    weights = {
        'lexical_overlap': 1,
        'tfidf': 2,
        'bow': 1,
        'ngram': 1,
        'char_level': 1,
        'pos': 1,
        'semantic_sbert': 3
    }

    total_weight = sum(weights.values())
    score = sum(features[k] * weights[k] for k in features) / total_weight
    return round(score * 100, 2), features

# Example usage:
if __name__ == "__main__":
    s1 = "I went to school today."
    s2 = "Today, I attended my classes at the school."
    score, details = sentence_similarity(s1, s2)
    print(f"Similarity Score: {score}%")
    print("Feature Breakdown:")
    for k, v in details.items():
        print(f"{k}: {round(v * 100, 2)}%")