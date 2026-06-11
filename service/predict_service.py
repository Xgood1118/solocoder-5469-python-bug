import logging

from predict.service import PredictionService
from predict.cluster import DefectClusterer
from config import CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

MAX_SIMILARITY_CHECK_RECORDS = 200


class PredictService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._prediction_service = PredictionService()
        self._clusterer = DefectClusterer()
        self._initialized = True

    def predict(self, description, model_type="svm", check_similarity=True):
        result = self._prediction_service.predict_single(description, model_type)

        if check_similarity:
            self._add_similarity_info(description, result)

        return result

    def _add_similarity_info(self, description, result):
        all_data = []
        try:
            from data.loader import load_data
            all_data = load_data()
        except Exception:
            logger.warning("Failed to load data for similarity check")

        if all_data:
            existing_descriptions = [r.get("description", "") for r in all_data[:MAX_SIMILARITY_CHECK_RECORDS]]
            try:
                is_similar, indices, scores = self._clusterer.is_similar_to_existing(
                    description, existing_descriptions, self._prediction_service._pipeline
                )
                if is_similar:
                    similar_items = []
                    for idx, score in zip(indices, scores):
                        if idx < len(all_data):
                            item = {
                                "index": idx,
                                "similarity": round(score, 4),
                                "category": all_data[idx].get("category", ""),
                                "severity": all_data[idx].get("severity", ""),
                                "description": all_data[idx].get("description", "")[:100],
                            }
                            similar_items.append(item)
                    result["similar_defects"] = similar_items
                    result["has_similar"] = True
                else:
                    result["similar_defects"] = []
                    result["has_similar"] = False
            except Exception as e:
                logger.warning("Similarity check failed: %s", e)
                result["similar_defects"] = []
                result["has_similar"] = False
        else:
            result["similar_defects"] = []
            result["has_similar"] = False

    def predict_batch(self, descriptions, model_type="svm"):
        return [self.predict(desc, model_type, check_similarity=False) for desc in descriptions]

    def get_review_queue(self):
        return self._prediction_service.get_review_queue()

    def set_cluster_threshold(self, threshold):
        self._clusterer.set_threshold(threshold)

    def check_similarity(self, description, existing_descriptions, pipeline):
        return self._clusterer.is_similar_to_existing(
            description, existing_descriptions, pipeline
        )
