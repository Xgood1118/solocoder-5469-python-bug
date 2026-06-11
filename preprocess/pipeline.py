import joblib
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import TFIDF_MAX_FEATURES, STOPWORDS_PATH
from preprocess.cleaner import clean_description
from preprocess.synonyms import merge_synonyms


def load_stopwords():
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def tokenize(text, stopwords=None):
    tokens = jieba.lcut(text)
    if stopwords:
        tokens = [t for t in tokens if t.strip() and t not in stopwords]
    tokens = merge_synonyms(tokens)
    return tokens


class PreprocessPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        self.stopwords = load_stopwords()

    def fit(self, texts):
        tokenized = [self._text_to_tfidf_input(t) for t in texts]
        self.vectorizer.fit(tokenized)
        return self

    def transform(self, texts):
        tokenized = [self._text_to_tfidf_input(t) for t in texts]
        return self.vectorizer.transform(tokenized)

    def fit_transform(self, texts):
        tokenized = [self._text_to_tfidf_input(t) for t in texts]
        return self.vectorizer.fit_transform(tokenized)

    def preprocess_single(self, text):
        cleaned_text, auxiliary_features = clean_description(text)
        tokens = tokenize(cleaned_text, self.stopwords)
        return tokens, auxiliary_features

    def get_auxiliary_feature_vector(self, auxiliary_features):
        binary_flags = [
            int(auxiliary_features.get("has_stack_trace", False)),
            int(auxiliary_features.get("has_api_path", False)),
            int(auxiliary_features.get("has_error_code", False)),
            int(auxiliary_features.get("has_json", False)),
            int(auxiliary_features.get("has_url", False)),
        ]
        api_path_count = len(auxiliary_features.get("api_path_parts", []))
        error_code_count = len(auxiliary_features.get("error_code_type", []))
        return np.array(binary_flags + [api_path_count, error_code_count], dtype=np.float64)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    def _text_to_tfidf_input(self, text):
        cleaned_text, _ = clean_description(text)
        tokens = tokenize(cleaned_text, self.stopwords)
        return " ".join(tokens)
