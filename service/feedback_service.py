import uuid
import datetime
import tempfile
import os
import json
import threading

from data.loader import load_data, save_data, add_records
from config import DATA_PATH


class FeedbackService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._write_lock = threading.Lock()
        self._initialized = True

    def submit_feedback(self, feedback_list):
        records = []
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for item in feedback_list:
            record = {
                "id": str(uuid.uuid4()),
                "description": item.get("description", ""),
                "category": item.get("category", ""),
                "severity": item.get("severity", ""),
                "confirmer": item.get("confirmer", ""),
                "confirmed": True,
                "timestamp": now,
            }
            records.append(record)

        with self._write_lock:
            add_records(records)

        return {"count": len(records), "ids": [r["id"] for r in records]}

    def _atomic_write_json(self, data, path):
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def get_feedback_stats(self):
        try:
            data = load_data()
        except Exception:
            return {"total_confirmed": 0, "category_breakdown": {}}

        confirmed = [r for r in data if r.get("confirmed")]
        category_breakdown = {}
        for r in confirmed:
            cat = r.get("category", "未知")
            category_breakdown[cat] = category_breakdown.get(cat, 0) + 1

        return {
            "total_confirmed": len(confirmed),
            "category_breakdown": category_breakdown,
        }

    def confirm_batch(self, prediction_ids, corrections=None):
        if corrections is None:
            corrections = {}

        with self._write_lock:
            data = load_data()
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            updated_count = 0

            for record in data:
                if record.get("id") in prediction_ids:
                    record["confirmed"] = True
                    record["timestamp"] = now
                    if record["id"] in corrections:
                        correction = corrections[record["id"]]
                        if "category" in correction:
                            record["category"] = correction["category"]
                        if "severity" in correction:
                            record["severity"] = correction["severity"]
                        if "confirmer" in correction:
                            record["confirmer"] = correction["confirmer"]
                    updated_count += 1

            self._atomic_write_json(data, DATA_PATH)

        return {"updated_count": updated_count}
