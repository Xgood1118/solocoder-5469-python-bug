import threading
import logging
import datetime
import os

import numpy as np

from data.loader import load_data, split_data
from model.trainer import ModelManager, train_all_models, NBTrainer, SVMTrainer
from evaluate.reporter import EvaluationReport, generate_evaluation_report, evaluate_model
from preprocess.pipeline import PreprocessPipeline
from config import RETRAIN_DAY, RETRAIN_HOUR, MODELS_DIR, CATEGORIES, SEVERITIES

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
            texts_train = [r["description"] for r in train_data]
            pipeline.fit(texts_train)

            manager = ModelManager()
            version = manager.increment_version()
            pipeline_path = os.path.join(MODELS_DIR, "pipeline_%s.joblib" % version)
            os.makedirs(MODELS_DIR, exist_ok=True)
            pipeline.save(pipeline_path)

            results = train_all_models(train_data, pipeline, version=version)

            report = EvaluationReport()

            X_test_texts = [r["description"] for r in test_data]
            y_test_cat = [r["category"] for r in test_data]
            y_test_sev = [r["severity"] for r in test_data]

            X_test_tfidf = pipeline.transform(X_test_texts)
            aux_test = np.array([
                pipeline.get_auxiliary_feature_vector(
                    pipeline.preprocess_single(t)[1]
                )
                for t in X_test_texts
            ])
            from scipy.sparse import hstack
            X_test_features = hstack([X_test_tfidf, aux_test])

            nb_cat_result = None
            nb_sev_result = None
            svm_cat_result = None
            svm_sev_result = None
            bert_cat_result = None
            bert_sev_result = None

            nb_cat_path = os.path.join(MODELS_DIR, "category_nb_%s.joblib" % version)
            if os.path.exists(nb_cat_path):
                nb_cat = NBTrainer.load(nb_cat_path)
                nb_cat_result = evaluate_model(
                    nb_cat, X_test_features, y_test_cat, CATEGORIES,
                    model_name="NB", task_name="category",
                )
                report.add_result("category", "NB", nb_cat_result)
                report.track_accuracy_over_time(
                    "NB", "category", nb_cat_result["accuracy"],
                    version, datetime.datetime.now().isoformat(),
                )

            nb_sev_path = os.path.join(MODELS_DIR, "severity_nb_%s.joblib" % version)
            if os.path.exists(nb_sev_path):
                nb_sev = NBTrainer.load(nb_sev_path)
                nb_sev_result = evaluate_model(
                    nb_sev, X_test_features, y_test_sev, SEVERITIES,
                    model_name="NB", task_name="severity",
                )
                report.add_result("severity", "NB", nb_sev_result)
                report.track_accuracy_over_time(
                    "NB", "severity", nb_sev_result["accuracy"],
                    version, datetime.datetime.now().isoformat(),
                )

            svm_cat_path = os.path.join(MODELS_DIR, "category_svm_%s.joblib" % version)
            if os.path.exists(svm_cat_path):
                svm_cat = SVMTrainer.load(svm_cat_path)
                svm_cat_result = evaluate_model(
                    svm_cat, X_test_features, y_test_cat, CATEGORIES,
                    model_name="SVM", task_name="category",
                )
                report.add_result("category", "SVM", svm_cat_result)
                report.track_accuracy_over_time(
                    "SVM", "category", svm_cat_result["accuracy"],
                    version, datetime.datetime.now().isoformat(),
                )

            svm_sev_path = os.path.join(MODELS_DIR, "severity_svm_%s.joblib" % version)
            if os.path.exists(svm_sev_path):
                svm_sev = SVMTrainer.load(svm_sev_path)
                svm_sev_result = evaluate_model(
                    svm_sev, X_test_features, y_test_sev, SEVERITIES,
                    model_name="SVM", task_name="severity",
                )
                report.add_result("severity", "SVM", svm_sev_result)
                report.track_accuracy_over_time(
                    "SVM", "severity", svm_sev_result["accuracy"],
                    version, datetime.datetime.now().isoformat(),
                )

            bert_cat_path = os.path.join(MODELS_DIR, "category_bert_%s.joblib" % version)
            if os.path.exists(bert_cat_path):
                try:
                    from model.bert_trainer import BERTTrainer
                    bert_cat = BERTTrainer.load(bert_cat_path)
                    bert_cat_result = evaluate_model(
                        bert_cat, X_test_texts, y_test_cat, CATEGORIES,
                        model_name="BERT", task_name="category", is_bert=True,
                    )
                    report.add_result("category", "BERT", bert_cat_result)
                    report.track_accuracy_over_time(
                        "BERT", "category", bert_cat_result["accuracy"],
                        version, datetime.datetime.now().isoformat(),
                    )
                except Exception as e:
                    logger.warning("BERT category evaluation failed: %s", e)

            bert_sev_path = os.path.join(MODELS_DIR, "severity_bert_%s.joblib" % version)
            if os.path.exists(bert_sev_path):
                try:
                    from model.bert_trainer import BERTTrainer
                    bert_sev = BERTTrainer.load(bert_sev_path)
                    bert_sev_result = evaluate_model(
                        bert_sev, X_test_texts, y_test_sev, SEVERITIES,
                        model_name="BERT", task_name="severity", is_bert=True,
                    )
                    report.add_result("severity", "BERT", bert_sev_result)
                    report.track_accuracy_over_time(
                        "BERT", "severity", bert_sev_result["accuracy"],
                        version, datetime.datetime.now().isoformat(),
                    )
                except Exception as e:
                    logger.warning("BERT severity evaluation failed: %s", e)

            report.save(os.path.join(MODELS_DIR, "evaluation_report.json"))

            if nb_cat_result is not None and nb_sev_result is not None and svm_cat_result is not None and svm_sev_result is not None:
                md_report_path = os.path.join(MODELS_DIR, "evaluation_report_%s.md" % version)
                generate_evaluation_report(
                    nb_cat_result, nb_sev_result,
                    svm_cat_result, svm_sev_result,
                    bert_cat_result=bert_cat_result,
                    bert_sev_result=bert_sev_result,
                    output_path=md_report_path,
                )
                logger.info("Evaluation report saved: %s", md_report_path)

            self._last_retrain_info = {
                "timestamp": datetime.datetime.now().isoformat(),
                "version": "v%s" % version,
                "n_train": len(train_data),
                "n_test": len(test_data),
                "results": {
                    "nb_category": nb_cat_result,
                    "nb_severity": nb_sev_result,
                    "svm_category": svm_cat_result,
                    "svm_severity": svm_sev_result,
                    "bert_category": bert_cat_result,
                    "bert_severity": bert_sev_result,
                },
            }

            logger.info("Retraining completed, version: v%s", version)

        except Exception as e:
            logger.error("Retraining failed: %s", e)
            import traceback
            logger.error(traceback.format_exc())
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
