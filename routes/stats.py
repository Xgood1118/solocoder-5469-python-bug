from flask import Blueprint, jsonify
from service.stats_service import StatsService

stats_bp = Blueprint('stats', __name__)

_stats_service = None


def _get_stats_service():
    global _stats_service
    if _stats_service is None:
        _stats_service = StatsService()
    return _stats_service


@stats_bp.route('/stats', methods=['GET'])
def get_stats():
    service = _get_stats_service()
    summary = service.get_stats_summary()
    charts = service.generate_all_charts()

    return jsonify({
        "summary": summary,
        "charts": {
            "category_pie": charts.get("category_pie"),
            "severity_bar": charts.get("severity_bar"),
            "monthly_trend": charts.get("monthly_trend"),
            "accuracy_trend": charts.get("accuracy_trend"),
        },
    })
