import os
import sys
import logging
import threading

from flask import Flask, send_from_directory
from config import PORT, HOST, STATIC_DIR, CHARTS_DIR, EXPORTS_DIR, MODELS_DIR
from data.loader import init_data_if_needed, load_data, split_data
from routes import register_blueprints
from service.retrain_service import RetrainService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _initial_train():
    version_path = os.path.join(MODELS_DIR, "version.txt")
    if os.path.exists(version_path):
        logger.info("Models already exist, skipping initial training")
        return

    logger.info("No trained models found, running initial training...")
    try:
        from preprocess.pipeline import PreprocessPipeline
        from model.trainer import train_all_models, ModelManager

        data = load_data()
        train_data, _ = split_data(data)

        pipeline = PreprocessPipeline()
        texts = [r["description"] for r in train_data]
        pipeline.fit(texts)

        manager = ModelManager()
        version = manager.increment_version()
        pipeline_path = os.path.join(MODELS_DIR, "pipeline_%s.joblib" % version)
        pipeline.save(pipeline_path)

        train_all_models(train_data, pipeline, version=version)
        logger.info("Initial training completed, version: %s", version)
    except Exception as e:
        logger.error("Initial training failed: %s", e)


def create_app():
    app = Flask(__name__, static_folder=STATIC_DIR)

    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    init_data_if_needed()

    train_thread = threading.Thread(target=_initial_train, daemon=True)
    train_thread.start()

    register_blueprints(app)

    @app.route("/charts/<path:filename>")
    def serve_chart(filename):
        return send_from_directory(CHARTS_DIR, filename)

    @app.route("/exports/<path:filename>")
    def serve_export(filename):
        return send_from_directory(EXPORTS_DIR, filename)

    @app.route("/health")
    def health():
        return {"status": "ok"}

    retrain_svc = RetrainService()
    retrain_svc.schedule_monthly_retrain()

    logger.info("Application initialized successfully")
    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("Starting server on %s:%s", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=False)
