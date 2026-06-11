import os

CATEGORIES = ["功能", "性能", "UI", "接口", "数据", "安全", "兼容性"]
SEVERITIES = ["致命", "严重", "一般", "轻微", "建议"]

PORT = int(os.environ.get("FLASK_PORT", 5000))
HOST = os.environ.get("FLASK_HOST", "0.0.0.0")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "defects.json")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
CHARTS_DIR = os.path.join(STATIC_DIR, "charts")
EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "exports")
STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), "preprocess", "stopwords.txt")

TFIDF_MAX_FEATURES = 8000
TEST_SIZE = 800
CONFIDENCE_THRESHOLD = 0.6
PREDICT_TIMEOUT_MS = 200
CLUSTER_THRESHOLD_DEFAULT = 0.9
CLUSTER_THRESHOLD_MIN = 0.85
CLUSTER_THRESHOLD_MAX = 0.95
BERT_MAX_LENGTH = 128
BERT_MODEL_NAME = "bert-base-chinese"

RETRAIN_DAY = 1
RETRAIN_HOUR = 2
