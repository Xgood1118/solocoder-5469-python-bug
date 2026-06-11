import threading
import logging
import datetime
import os

from data.loader import load_data, split_data
from model.trainer import ModelManager, train_all_models
from evaluate.reporter import EvaluationReport
from preprocess.pipeline import PreprocessPipeline
from config import RETRAIN_DAY, RETRAIN_HOUR, MODELS_DIR

logger = logging.getLogger(__name__)


class RetrainService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._retrain_lock = threading.Lock()
        self._is_retraining = False
        self._last_retrain_info = None
        self._scheduler_timer = None
        self._initialized = True

    def trigger_retrain(self):
        with self._retrain_lock:
            if self._is_retraining:
                return {"status": "already_retraining"}
            self._is_retraining = True

        thread = threading.Thread(target=self._do_retrain, daemon=True)
        thread.start()
        return {"status": "started"}

    def _do_retrain(self):
        try:
            logger.info("Starting retraining process")
            data = load_data()

            train_data, test_data = split_data(data)

            pipeline = PreprocessPipeline()
            texts = [r["description"] for r in train_data]
            pipeline.fit(texts)

            manager = ModelManager()
            version = manager.increment_version()
            pipeline_path = os.path.join(MODELS_DIR, f"pipeline_{version}.joblib")
            os.makedirs(MODELS_DIR, exist_ok=True)
            pipeline.save(pipeline_path)

            results = train_all_models(train_data, pipeline, version=version)

            report = EvaluationReport()

            for task in ["category", "severity"]:
                for model_name in ["nb", "svm"]:
                    key = f"{task}_{model_name}"
                    if key in results:
                        info = results[key]
                        report.add_result(task, model_name, {
                            "accuracy": info["accuracy"],
                            "f1_weighted": info.get("avg_confidence", 0),
                            "f1_macro": info.get("avg_confidence", 0),
                        })
                        report.track_accuracy_over_time(
                            model_name, task, info["accuracy"],
                            version, info.get("trained_at"),
                        )

            report_path = os.path.join(MODELS_DIR, "evaluation_report.json")
            report.save(report_path)

            self._last_retrain_info = {
                "timestamp": datetime.datetime.now().isoformat(),
                "version": f"v{version}",
                "n_train": len(train_data),
                "n_test": len(test_data),
                "results": results,
            }

            logger.info("Retraining completed, version: v%s", version)

        except Exception as e:
            logger.error("Retraining failed: %s", e)
            self._last_retrain_info = {
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e),
            }
        finally:
            with self._retrain_lock:
                self._is_retraining = False

    def is_retraining(self):
        with self._retrain_lock:
            return self._is_retraining

    def get_last_retrain_info(self):
        return self._last_retrain_info

    def schedule_monthly_retrain(self):
        def _calc_next_delay():
            now = datetime.datetime.now()
            if now.day < RETRAIN_DAY:
                next_run = now.replace(day=RETRAIN_DAY, hour=RETRAIN_HOUR, minute=0, second=0, microsecond=0)
            else:
                if now.month == 12:
                    next_run = now.replace(year=now.year + 1, month=1, day=RETRAIN_DAY, hour=RETRAIN_HOUR, minute=0, second=0, microsecond=0)
                else:
                    next_run = now.replace(month=now.month + 1, day=RETRAIN_DAY, hour=RETRAIN_HOUR, minute=0, second=0, microsecond=0)
            delta = next_run - now
            return max(delta.total_seconds(), 60)

        def _scheduled_retrain():
            self.trigger_retrain()
            self.schedule_monthly_retrain()

        delay = _calc_next_delay()
        logger.info("Next scheduled retrain in %.0f seconds", delay)
        self._scheduler_timer = threading.Timer(delay, _scheduled_retrain)
        self._scheduler_timer.daemon = True
        self._scheduler_timer.start()
