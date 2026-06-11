from flask import Blueprint, request, jsonify
from service.predict_service import PredictService

cluster_bp = Blueprint('cluster', __name__)

_predict_service = None


def _get_predict_service():
    global _predict_service
    if _predict_service is None:
        _predict_service = PredictService()
    return _predict_service


@cluster_bp.route('/cluster/threshold', methods=['POST'])
def set_cluster_threshold():
    data = request.get_json()
    if not data or 'threshold' not in data:
        return jsonify({"error": "缺少必填字段: threshold"}), 400

    threshold = data['threshold']
    try:
        threshold = float(threshold)
        service = _get_predict_service()
        service.set_cluster_threshold(threshold)
        return jsonify({"threshold": threshold, "message": "聚类阈值更新成功"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@cluster_bp.route('/cluster/check', methods=['POST'])
def check_similarity():
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({"error": "缺少必填字段: description"}), 400

    description = data['description']
    service = _get_predict_service()

    try:
        from data.loader import load_data
        all_data = load_data()
        existing = [r.get("description", "") for r in all_data]
        pipeline = service._prediction_service._pipeline
        is_similar, indices, scores = service.check_similarity(description, existing, pipeline)

        similar_items = []
        for idx, score in zip(indices, scores):
            if idx < len(all_data):
                similar_items.append({
                    "index": idx,
                    "similarity": round(score, 4),
                    "category": all_data[idx].get("category", ""),
                    "description": all_data[idx].get("description", "")[:100],
                })

        return jsonify({
            "is_similar": is_similar,
            "similar_count": len(similar_items),
            "similar_defects": similar_items,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
