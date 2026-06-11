import threading
import uuid
import time
import json
from collections import OrderedDict

from service.predict_service import PredictService


class BatchTaskManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._tasks = OrderedDict()
        self._max_tasks = 100
        self._lock = threading.Lock()
        self._initialized = True

    def submit_task(self, descriptions, model_type="svm"):
        with self._lock:
            self.cleanup_old_tasks()
            if len(self._tasks) >= self._max_tasks:
                oldest_key = next(iter(self._tasks))
                del self._tasks[oldest_key]

            task_id = str(uuid.uuid4())
            task = {
                "task_id": task_id,
                "status": "pending",
                "progress": 0,
                "total": len(descriptions),
                "results": [],
                "created_at": time.time(),
                "completed_at": None,
                "descriptions": descriptions,
                "model_type": model_type,
            }
            self._tasks[task_id] = task

        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, descriptions, model_type),
            daemon=True,
        )
        thread.start()
        return task_id

    def _run_task(self, task_id, descriptions, model_type):
        with self._lock:
            if task_id not in self._tasks:
                return
            self._tasks[task_id]["status"] = "running"

        predict_service = PredictService()

        try:
            results = []
            for i, desc in enumerate(descriptions):
                try:
                    result = predict_service.predict(desc, model_type)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "description": desc})

                with self._lock:
                    if task_id not in self._tasks:
                        return
                    self._tasks[task_id]["progress"] = i + 1
                    self._tasks[task_id]["results"] = results

            with self._lock:
                if task_id not in self._tasks:
                    return
                self._tasks[task_id]["status"] = "completed"
                self._tasks[task_id]["completed_at"] = time.time()

        except Exception as e:
            with self._lock:
                if task_id not in self._tasks:
                    return
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["error"] = str(e)
                self._tasks[task_id]["completed_at"] = time.time()

    def get_task_status(self, task_id):
        with self._lock:
            if task_id not in self._tasks:
                return None
            task = self._tasks[task_id]
            return {
                "task_id": task["task_id"],
                "status": task["status"],
                "progress": task["progress"],
                "total": task["total"],
                "results": task["results"] if task["status"] == "completed" else [],
            }

    def get_task_result(self, task_id):
        with self._lock:
            if task_id not in self._tasks:
                return None
            task = self._tasks[task_id]
            if task["status"] != "completed":
                return None
            return {
                "task_id": task["task_id"],
                "status": task["status"],
                "progress": task["progress"],
                "total": task["total"],
                "results": task["results"],
                "created_at": task["created_at"],
                "completed_at": task["completed_at"],
            }

    def cleanup_old_tasks(self):
        now = time.time()
        expired_keys = []
        for key, task in self._tasks.items():
            if task["status"] in ("completed", "failed") and task["completed_at"]:
                if now - task["completed_at"] > 3600:
                    expired_keys.append(key)
        for key in expired_keys:
            del self._tasks[key]

    def get_task_stream(self, task_id):
        while True:
            with self._lock:
                if task_id not in self._tasks:
                    yield f"data: {json.dumps({'error': 'task not found'})}\n\n"
                    return
                task = self._tasks[task_id]
                status_data = {
                    "task_id": task["task_id"],
                    "status": task["status"],
                    "progress": task["progress"],
                    "total": task["total"],
                }
                if task["status"] == "completed":
                    status_data["results"] = task["results"]

            yield f"data: {json.dumps(status_data)}\n\n"

            if task["status"] in ("completed", "failed"):
                return

            time.sleep(0.5)
