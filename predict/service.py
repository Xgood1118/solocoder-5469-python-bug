import os
import time
import logging

import joblib
import numpy as np

from config import CATEGORIES, SEVERITIES, CONFIDENCE_THRESHOLD, MODELS_DIR, PREDICT_TIMEOUT_MS
from preprocess.pipeline import PreprocessPipeline

logger = logging.getLogger(__name__)

MODEL_NAMES = ["nb", "svm", "bert"]


class PredictionService:
    def __init__(self):
        self._models = {}
        self._mtimes = {}
        self._pipeline = None
        self._pipeline_mtime = None
        self._version = None
        self._review_queue = []
        self._load_models()

    def _load_models(self):
        version_path = os.path.join(MODELS_DIR, "version.txt")
        if os.path.exists(version_path):
            with open(version_path, "r") as f:
                self._version = f.read().strip()
        else:
            self._version = "1"

        pipeline_path = os.path.join(MODELS_DIR, f"pipeline_v{self._version}.joblib")
        if os.path.exists(pipeline_path):
            self._pipeline = joblib.load(pipeline_path)
            self._pipeline_mtime = os.path.getmtime(pipeline_path)
        else:
            fallback = os.path.join(MODELS_DIR, "pipeline.pkl")
            if os.path.exists(fallback):
                self._pipeline = joblib.load(fallback)
                self._pipeline_mtime = os.path.getmtime(fallback)
            else:
                self._pipeline = PreprocessPipeline()
                logger.warning("Pipeline file not found, using fresh instance")

        self._models = {}
        self._mtimes = {}
        for task in ["category", "severity"]:
            for model_type in MODEL_NAMES:
                filename = f"{task}_{model_type}_v{self._version}.joblib"
                filepath = os.path.join(MODELS_DIR, filename)
                if not os.path.exists(filepath):
                    fallback = os.path.join(MODELS_DIR, f"{task}_{model_type}.pkl")
                    if os.path.exists(fallback):
                        filepath = fallback
                    else:
                        if model_type != "bert":
                            logger.warning("Model file not found: %s", filename)
                        continue
                self._models[(task, model_type)] = self._load_model(filepath, model_type)
                self._mtimes[(task, model_type)] = os.path.getmtime(filepath)

    @staticmethod
    def _load_model(filepath, model_type):
        if model_type == "bert":
            from model.bert_trainer import BERTTrainer
            return BERTTrainer.load(filepath)
        else:
            return joblib.load(filepath)

    def _check_hot_reload(self):
        version_path = os.path.join(MODELS_DIR, "version.txt")
        if os.path.exists(version_path):
            with open(version_path, "r") as f:
                new_version = f.read().strip()
            if new_version != self._version:
                logger.info("Model version changed: %s -> %s, reloading", self._version, new_version)
                self._version = new_version
                self._load_models()
                return

        for key, old_mtime in list(self._mtimes.items()):
            task, model_type = key
            filename = f"{task}_{model_type}_v{self._version}.joblib"
            filepath = os.path.join(MODELS_DIR, filename)
            if not os.path.exists(filepath):
                fallback = os.path.join(MODELS_DIR, f"{task}_{model_type}.pkl")
                if os.path.exists(fallback):
                    filepath = fallback
                else:
                    continue
            current_mtime = os.path.getmtime(filepath)
            if current_mtime != old_mtime:
                logger.info("Hot-reloading model: %s", filename)
                self._models[key] = self._load_model(filepath, model_type)
                self._mtimes[key] = current_mtime

    @staticmethod
    def _get_confidence(cat_conf, sev_conf):
        return min(cat_conf, sev_conf)

    def predict_single(self, description, model_type="svm"):
        self._check_hot_reload()

        start_ms = time.time() * 1000

        cat_key = ("category", model_type)
        sev_key = ("severity", model_type)

        if cat_key not in self._models or sev_key not in self._models:
            raise ValueError("Model type '%s' not available" % model_type)

        cat_trainer = self._models[cat_key]
        sev_trainer = self._models[sev_key]

        if model_type == "bert":
            cat_pred_labels, cat_conf_scores = cat_trainer.predict(description)
            sev_pred_labels, sev_conf_scores = sev_trainer.predict(description)
        else:
            tokens, auxiliary_features = self._pipeline.preprocess_single(description)
            tfidf_input = [" ".join(tokens)]
            tfidf_vec = self._pipeline.transform(tfidf_input)
            aux_vec = self._pipeline.get_auxiliary_feature_vector(auxiliary_features).reshape(1, -1)

            from scipy.sparse import hstack
            feature_vec = hstack([tfidf_vec, aux_vec])

            cat_pred_labels, cat_conf_scores = cat_trainer.predict(feature_vec)
            sev_pred_labels, sev_conf_scores = sev_trainer.predict(feature_vec)

        cat_label = str(cat_pred_labels[0])
        sev_label = str(sev_pred_labels[0])
        cat_conf = float(cat_conf_scores[0])
        sev_conf = float(sev_conf_scores[0])
        confidence = self._get_confidence(cat_conf, sev_conf)

        predict_time_ms = time.time() * 1000 - start_ms

        needs_review = confidence < CONFIDENCE_THRESHOLD

        if predict_time_ms > PREDICT_TIMEOUT_MS:
            logger.warning(
                "Prediction time %.1fms exceeded threshold %dms",
                predict_time_ms, PREDICT_TIMEOUT_MS,
            )

        result = {
            "category": cat_label,
            "severity": sev_label,
            "category_confidence": round(cat_conf, 4),
            "severity_confidence": round(sev_conf, 4),
            "confidence": round(confidence, 4),
            "needs_review": needs_review,
            "model_version": "v%s" % self._version,
            "predict_time_ms": round(predict_time_ms, 2),
            "model_type": model_type,
        }

        if needs_review:
            self.add_to_review_queue(result)

        return result

    def predict_batch(self, descriptions, model_type="svm"):
        return [self.predict_single(desc, model_type) for desc in descriptions]

    def get_review_queue(self):
        return list(self._review_queue)

    def add_to_review_queue(self, prediction):
        self._review_queue.append(prediction)
