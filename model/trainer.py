import os
import datetime
import logging

import joblib
import numpy as np
import scipy.sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from config import CATEGORIES, SEVERITIES, MODELS_DIR


class ModelManager:
    VERSION_FILE = "version.txt"

    def __init__(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        self._mtimes = {}
        self._loaded_models = {}

    def get_current_version(self):
        path = os.path.join(MODELS_DIR, self.VERSION_FILE)
        if not os.path.exists(path):
            return 0
        with open(path, "r") as f:
            content = f.read().strip()
            if not content:
                return 0
            return int(content)

    def increment_version(self):
        new_ver = self.get_current_version() + 1
        path = os.path.join(MODELS_DIR, self.VERSION_FILE)
        with open(path, "w") as f:
            f.write(str(new_ver))
        return f"v{new_ver}"

    def get_model_path(self, task, version):
        return os.path.join(MODELS_DIR, f"{task}_{version}.joblib")

    def check_mtime_and_reload(self, task, version):
        model_path = self.get_model_path(task, version)
        if not os.path.exists(model_path):
            return None

        current_mtime = os.path.getmtime(model_path)
        cache_key = f"{task}_{version}"

        if self._mtimes.get(cache_key) == current_mtime:
            return self._loaded_models.get(cache_key)

        model = joblib.load(model_path)
        self._mtimes[cache_key] = current_mtime
        self._loaded_models[cache_key] = model
        return model


def _build_class_weight_dict(y_labels):
    unique, counts = np.unique(y_labels, return_counts=True)
    total = len(y_labels)
    weight_map = {}
    for label, count in zip(unique, counts):
        weight_map[label] = total / (len(unique) * count)
    return weight_map


def _combine_features(X_tfidf, auxiliary_features):
    if auxiliary_features is None:
        return X_tfidf
    aux = np.array(auxiliary_features, dtype=np.float64)
    if aux.ndim == 1:
        aux = aux.reshape(1, -1)
    if X_tfidf.ndim == 1:
        X_tfidf = X_tfidf.reshape(1, -1)
    if scipy.sparse.issparse(X_tfidf):
        aux_sparse = scipy.sparse.csr_matrix(aux)
        return scipy.sparse.hstack([X_tfidf, aux_sparse])
    return np.hstack([X_tfidf, aux])


class NBTrainer:
    def __init__(self, task):
        if task not in ("category", "severity"):
            raise ValueError(f"task must be 'category' or 'severity', got '{task}'")
        self.task = task
        self.model = None
        self.classes_ = None

    def train(self, X_tfidf, y_labels, auxiliary_features=None):
        X = _combine_features(X_tfidf, auxiliary_features)
        sample_weights = compute_sample_weight("balanced", y_labels)
        self.model = MultinomialNB()
        self.model.fit(X, y_labels, sample_weight=sample_weights)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X_tfidf, auxiliary_features=None):
        X = _combine_features(X_tfidf, auxiliary_features)
        proba = self.model.predict_proba(X)
        predicted_indices = np.argmax(proba, axis=1)
        predicted_labels = np.array([self.classes_[i] for i in predicted_indices])
        confidence_scores = np.max(proba, axis=1)
        return predicted_labels, confidence_scores

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class SVMTrainer:
    def __init__(self, task):
        if task not in ("category", "severity"):
            raise ValueError(f"task must be 'category' or 'severity', got '{task}'")
        self.task = task
        self.model = None
        self.classes_ = None

    def train(self, X_tfidf, y_labels, auxiliary_features=None):
        X = _combine_features(X_tfidf, auxiliary_features)
        class_weight_dict = _build_class_weight_dict(y_labels)
        self.model = LinearSVC(class_weight=class_weight_dict, max_iter=10000)
        self.model.fit(X, y_labels)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X_tfidf, auxiliary_features=None):
        X = _combine_features(X_tfidf, auxiliary_features)
        decision = self.model.decision_function(X)
        if decision.ndim == 1:
            exp_vals = np.exp(decision - np.max(decision))
            proba = exp_vals / exp_vals.sum()
            predicted_labels = np.array([
                self.classes_[0] if d >= 0 else self.classes_[1]
                for d in decision
            ])
            confidence_scores = np.maximum(proba, 1 - proba)
        else:
            exp_vals = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            proba = exp_vals / exp_vals.sum(axis=1, keepdims=True)
            predicted_indices = np.argmax(proba, axis=1)
            predicted_labels = np.array([self.classes_[i] for i in predicted_indices])
            confidence_scores = np.max(proba, axis=1)
        return predicted_labels, confidence_scores

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


def train_all_models(train_data, preprocess_pipeline, version=None, include_bert=True):
    manager = ModelManager()
    if version is None:
        version = manager.increment_version()
    version_str = "v%d" % version if isinstance(version, int) else version

    texts = [r["description"] for r in train_data]
    category_labels = np.array([r["category"] for r in train_data])
    severity_labels = np.array([r["severity"] for r in train_data])

    X_tfidf = preprocess_pipeline.fit_transform(texts)

    auxiliary_matrix = np.array([
        preprocess_pipeline.get_auxiliary_feature_vector(
            preprocess_pipeline.preprocess_single(t)[1]
        )
        for t in texts
    ])

    results = {}

    for task, labels, label_set in [
        ("category", category_labels, CATEGORIES),
        ("severity", severity_labels, SEVERITIES),
    ]:
        for trainer_cls in (NBTrainer, SVMTrainer):
            trainer_name = trainer_cls.__name__.replace("Trainer", "").lower()
            trainer = trainer_cls(task)
            trainer.train(X_tfidf, labels, auxiliary_matrix)

            model_path = manager.get_model_path(f"{task}_{trainer_name}", version_str)
            trainer.save(model_path)

            pred_labels, conf_scores = trainer.predict(X_tfidf, auxiliary_matrix)
            accuracy = np.mean(pred_labels == labels)
            avg_confidence = float(np.mean(conf_scores))

            results[f"{task}_{trainer_name}"] = {
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "model_path": model_path,
                "version": version_str,
                "trained_at": datetime.datetime.now().isoformat(),
                "n_samples": len(labels),
            }

    if include_bert:
        try:
            from model.bert_trainer import train_bert_models
            bert_results = train_bert_models(train_data, version_str)
            results.update(bert_results)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("BERT training skipped or failed: %s", e)

    return results
