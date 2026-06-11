import json
from flask import Blueprint, jsonify, Response
from service.batch_service import BatchTaskManager

tasks_bp = Blueprint('tasks', __name__)

_batch_manager = None


def _get_batch_manager():
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = BatchTaskManager()
    return _batch_manager


@tasks_bp.route('/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    manager = _get_batch_manager()
    task = manager.get_task_status(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404

    response = {
        "task_id": task.get("task_id"),
        "status": task.get("status"),
        "progress": task.get("progress", 0),
        "total": task.get("total", 0),
    }
    if task.get("results") is not None:
        response["results"] = task["results"]
    return jsonify(response)


@tasks_bp.route('/tasks/<task_id>/stream', methods=['GET'])
def stream_task(task_id):
    manager = _get_batch_manager()

    def generate():
        for event in manager.get_task_stream(task_id):
            yield event

    return Response(generate(), mimetype='text/event-stream')
