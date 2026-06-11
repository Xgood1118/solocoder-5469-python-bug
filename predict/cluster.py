import logging

import numpy as np
from scipy.sparse import issparse, vstack
from sklearn.metrics.pairwise import cosine_similarity

from config import CLUSTER_THRESHOLD_DEFAULT, CLUSTER_THRESHOLD_MIN, CLUSTER_THRESHOLD_MAX

logger = logging.getLogger(__name__)


class DefectClusterer:
    def __init__(self, threshold=CLUSTER_THRESHOLD_DEFAULT):
        self._validate_threshold(threshold)
        self.threshold = threshold

    @staticmethod
    def _validate_threshold(threshold):
        if not CLUSTER_THRESHOLD_MIN <= threshold <= CLUSTER_THRESHOLD_MAX:
            raise ValueError(
                f"Threshold {threshold} out of range "
                f"[{CLUSTER_THRESHOLD_MIN}, {CLUSTER_THRESHOLD_MAX}]"
            )

    def set_threshold(self, threshold):
        self._validate_threshold(threshold)
        self.threshold = threshold

    @staticmethod
    def compute_similarity(text1_tfidf, text2_tfidf):
        if issparse(text1_tfidf):
            text1_tfidf = text1_tfidf.toarray()
        if issparse(text2_tfidf):
            text2_tfidf = text2_tfidf.toarray()
        return float(cosine_similarity(text1_tfidf, text2_tfidf)[0][0])

    def find_similar(self, description_tfidf, all_tfidf_matrix, threshold=None):
        effective_threshold = threshold if threshold is not None else self.threshold
        similarities = cosine_similarity(description_tfidf, all_tfidf_matrix)[0]
        results = []
        for idx, score in enumerate(similarities):
            if score >= effective_threshold:
                results.append((idx, float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def cluster_defects(self, tfidf_matrix, descriptions=None):
        n_samples = tfidf_matrix.shape[0]
        similarity_matrix = cosine_similarity(tfidf_matrix)

        visited = [False] * n_samples
        clusters = []

        for i in range(n_samples):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            queue = [i]
            while queue:
                current = queue.pop(0)
                for j in range(n_samples):
                    if not visited[j] and similarity_matrix[current][j] >= self.threshold:
                        cluster.append(j)
                        visited[j] = True
                        queue.append(j)
            clusters.append(cluster)

        return clusters

    def is_similar_to_existing(self, new_description, existing_descriptions, pipeline, threshold=None):
        effective_threshold = threshold if threshold is not None else self.threshold
        new_tokens, _ = pipeline.preprocess_single(new_description)
        new_text = " ".join(new_tokens)

        all_texts = []
        for desc in existing_descriptions:
            tokens, _ = pipeline.preprocess_single(desc)
            all_texts.append(" ".join(tokens))
        all_texts.append(new_text)

        tfidf_matrix = pipeline.transform(all_texts)
        new_vec = tfidf_matrix[-1:]
        existing_matrix = tfidf_matrix[:-1]

        similar_indices = []
        scores = []
        similarities = cosine_similarity(new_vec, existing_matrix)[0]

        for idx, score in enumerate(similarities):
            if score >= effective_threshold:
                similar_indices.append(idx)
                scores.append(float(score))

        is_similar = len(similar_indices) > 0
        return is_similar, similar_indices, scores
