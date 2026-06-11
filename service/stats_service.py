import collections
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CATEGORIES, SEVERITIES, CHARTS_DIR, DATA_PATH
from data.loader import load_data


def _setup_chinese_font():
    for font_name in ["SimHei", "Microsoft YaHei", "STSong", "Arial Unicode MS"]:
        try:
            from matplotlib.font_manager import FontProperties
            fp = FontProperties(family=font_name)
            if fp.get_name() != font_name and font_name != "SimHei":
                continue
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


class StatsService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

    def generate_all_charts(self):
        paths = {
            "category_pie": self.generate_category_pie(),
            "severity_bar": self.generate_severity_bar(),
            "monthly_trend": self.generate_monthly_trend(),
            "accuracy_trend": self.generate_accuracy_trend(),
        }
        return paths

    def generate_category_pie(self):
        data = load_data()
        cat_counts = collections.Counter(r.get("category", "未知") for r in data)

        labels = []
        sizes = []
        for cat in CATEGORIES:
            if cat in cat_counts:
                labels.append(cat)
                sizes.append(cat_counts[cat])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("缺陷类别分布")
        plt.tight_layout()

        os.makedirs(CHARTS_DIR, exist_ok=True)
        output_path = os.path.join(CHARTS_DIR, "category_pie.png")
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        return output_path

    def generate_severity_bar(self):
        data = load_data()
        sev_counts = collections.Counter(r.get("severity", "未知") for r in data)

        labels = []
        counts = []
        for sev in SEVERITIES:
            if sev in sev_counts:
                labels.append(sev)
                counts.append(sev_counts[sev])

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels, counts, color=plt.cm.Set2.colors[: len(labels)])
        ax.set_xlabel("严重程度")
        ax.set_ylabel("数量")
        ax.set_title("缺陷严重程度分布")

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        os.makedirs(CHARTS_DIR, exist_ok=True)
        output_path = os.path.join(CHARTS_DIR, "severity_bar.png")
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        return output_path

    def generate_monthly_trend(self):
        data = load_data()
        monthly_counts = collections.OrderedDict()

        for r in data:
            ts = r.get("timestamp", "")
            if len(ts) >= 7:
                month_key = ts[:7]
                monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1

        sorted_months = sorted(monthly_counts.keys())
        counts = [monthly_counts[m] for m in sorted_months]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_months, counts, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("月份")
        ax.set_ylabel("新增缺陷数")
        ax.set_title("月度新增缺陷趋势")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        os.makedirs(CHARTS_DIR, exist_ok=True)
        output_path = os.path.join(CHARTS_DIR, "monthly_trend.png")
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        return output_path

    def generate_accuracy_trend(self):
        report_dir = os.path.join(os.path.dirname(DATA_PATH), "..", "models")
        eval_path = os.path.join(report_dir, "evaluation_report.json")

        records = []
        if os.path.exists(eval_path):
            import json
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            history = eval_data.get("accuracy_history", {})
            for key, entries in history.items():
                for entry in entries:
                    records.append({
                        "label": key,
                        "version": entry.get("version", ""),
                        "accuracy": entry.get("accuracy", 0),
                        "timestamp": entry.get("timestamp", ""),
                    })

        if not records:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "暂无准确率数据", ha="center", va="center", fontsize=14)
            ax.set_title("模型准确率趋势")
            plt.tight_layout()
            os.makedirs(CHARTS_DIR, exist_ok=True)
            output_path = os.path.join(CHARTS_DIR, "accuracy_trend.png")
            fig.savefig(output_path, dpi=100)
            plt.close(fig)
            return output_path

        fig, ax = plt.subplots(figsize=(10, 6))
        grouped = collections.OrderedDict()
        for r in records:
            label = r["label"]
            if label not in grouped:
                grouped[label] = {"timestamps": [], "accuracies": []}
            grouped[label]["timestamps"].append(r["timestamp"])
            grouped[label]["accuracies"].append(r["accuracy"])

        for label, group in grouped.items():
            ax.plot(
                group["timestamps"],
                group["accuracies"],
                marker="o",
                label=label,
                linewidth=2,
            )

        ax.set_xlabel("时间")
        ax.set_ylabel("准确率")
        ax.set_title("模型准确率趋势")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        os.makedirs(CHARTS_DIR, exist_ok=True)
        output_path = os.path.join(CHARTS_DIR, "accuracy_trend.png")
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
        return output_path

    def get_stats_summary(self):
        data = load_data()

        cat_counts = collections.Counter(r.get("category", "未知") for r in data)
        sev_counts = collections.Counter(r.get("severity", "未知") for r in data)

        monthly_counts = collections.OrderedDict()
        for r in data:
            ts = r.get("timestamp", "")
            if len(ts) >= 7:
                month_key = ts[:7]
                monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1

        review_queue = []
        try:
            from service.predict_service import PredictService
            review_queue = PredictService().get_review_queue()
        except Exception:
            pass

        return {
            "total": len(data),
            "category_counts": dict(cat_counts),
            "severity_counts": dict(sev_counts),
            "monthly_counts": dict(sorted(monthly_counts.items())),
            "review_queue_size": len(review_queue),
        }
