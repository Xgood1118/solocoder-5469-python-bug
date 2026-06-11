from flask import Blueprint, request, jsonify
from service.feedback_service import FeedbackService

feedback_bp = Blueprint('feedback', __name__)

_feedback_service = None


def _get_feedback_service():
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService()
    return _feedback_service


@feedback_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体不能为空"}), 400

    required_fields = ['description', 'category', 'severity', 'confirmer']

    if 'feedbacks' in data:
        feedbacks = data['feedbacks']
        if not isinstance(feedbacks, list) or len(feedbacks) == 0:
            return jsonify({"error": "feedbacks 必须是非空数组"}), 400

        for fb in feedbacks:
            missing = [f for f in required_fields if f not in fb]
            if missing:
                return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    else:
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400

        feedbacks = [{
            "description": data["description"],
            "category": data["category"],
            "severity": data["severity"],
            "confirmer": data["confirmer"],
        }]

    service = _get_feedback_service()
    result = service.submit_feedback(feedbacks)

    return jsonify({
        "accepted_count": result.get("count", 0),
        "message": f"成功接受 {result.get('count', 0)} 条反馈",
    })
