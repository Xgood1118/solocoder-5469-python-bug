import time
import json
from flask import Blueprint, request, jsonify, Response
from service.predict_service import PredictService
from service.batch_service import BatchTaskManager

predict_bp = Blueprint('predict', __name__)

_predict_service = None
_batch_manager = None


def _get_predict_service():
    global _predict_service
    if _predict_service is None:
        _predict_service = PredictService()
    return _predict_service


def _get_batch_manager():
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = BatchTaskManager()
    return _batch_manager


@predict_bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({"error": "缺少必填字段: description"}), 400

    description = data['description']
    model_type = data.get('model_type', 'svm')

    start = time.time()
    service = _get_predict_service()
    result = service.predict(description, model_type=model_type)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    response = {
        "category": result.get("category"),
        "severity": result.get("severity"),
        "category_confidence": result.get("category_confidence"),
        "severity_confidence": result.get("severity_confidence"),
        "confidence": result.get("confidence"),
        "needs_review": result.get("needs_review"),
        "model_version": result.get("model_version"),
        "predict_time_ms": elapsed_ms,
        "similar_defects": result.get("similar_defects", []),
        "has_similar": result.get("has_similar", False),
    }
    return jsonify(response)


@predict_bp.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.get_json()
    if not data or 'descriptions' not in data:
        return jsonify({"error": "缺少必填字段: descriptions"}), 400

    descriptions = data['descriptions']
    model_type = data.get('model_type', 'svm')
    mode = data.get('mode', 'polling')

    manager = _get_batch_manager()

    if mode == 'sse':
        task_id = manager.submit_task(descriptions, model_type=model_type)

        def generate():
            for event in manager.get_task_stream(task_id):
                yield event

        return Response(generate(), mimetype='text/event-stream')

    task_id = manager.submit_task(descriptions, model_type=model_type)
    return jsonify({
        "task_id": task_id,
        "status": "pending",
    })
