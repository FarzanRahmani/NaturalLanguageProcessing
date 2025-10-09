import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from nltk import pos_tag, word_tokenize
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer, util
import hazm

# Initialize Persian NLP tools
normalizer = hazm.Normalizer()
tokenizer = hazm.WordTokenizer()
stopwords = set(hazm.stopwords_list())
lemmatizer = hazm.Lemmatizer()
sbert_model = SentenceTransformer('HooshvareLab/bert-base-parsbert-uncased-sentiment')

def preprocess(text):
    text = normalizer.normalize(text)
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords and token.strip()]
    return tokens

def lexical_overlap(s1, s2):
    t1, t2 = preprocess(s1), preprocess(s2)
    overlap = len(set(t1).intersection(set(t2))) / max(len(set(t1).union(set(t2))), 1)
    return overlap

def tfidf_similarity(s1, s2):
    vect = TfidfVectorizer(tokenizer=preprocess).fit([s1, s2])
    vecs = vect.transform([s1, s2])
    return cosine_similarity(vecs[0], vecs[1])[0][0]

def bow_similarity(s1, s2):
    vect = CountVectorizer(tokenizer=preprocess).fit([s1, s2])
    vecs = vect.transform([s1, s2])
    return cosine_similarity(vecs[0], vecs[1])[0][0]

def ngram_overlap(s1, s2, n=2):
    t1, t2 = preprocess(s1), preprocess(s2)
    n1 = set(ngrams(t1, n))
    n2 = set(ngrams(t2, n))
    return len(n1.intersection(n2)) / max(len(n1.union(n2)), 1)

def char_level_similarity(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

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
        'semantic_sbert': semantic_similarity_sbert(s1, s2)
    }

    weights = {
        'lexical_overlap': 1,
        'tfidf': 2,
        'bow': 1,
        'ngram': 1,
        'char_level': 1,
        'semantic_sbert': 4
    }

    total_weight = sum(weights.values())
    score = sum(features[k] * weights[k] for k in features) / total_weight
    return round(score * 100, 2), features

# Example usage:
if __name__ == "__main__":
    s1 = "من امروز به مدرسه رفتم."
    s2 = "امروز به دبیرستان رفته بودم."
    score, details = sentence_similarity(s1, s2)
    print(f"درصد شباهت: {score}%")
    print("جزئیات ویژگی‌ها:")
    for k, v in details.items():
        print(f"{k}: {round(v * 100, 2)}%")