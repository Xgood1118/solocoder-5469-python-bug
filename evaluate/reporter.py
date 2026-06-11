from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from config import CATEGORIES, SEVERITIES
from datetime import datetime
import os
import json


def evaluate_model(model, X_test, y_test, label_names, model_name, task_name, is_bert=False):
    if is_bert:
        pred_result = model.predict(X_test)
    else:
        pred_result = model.predict(X_test)
    if isinstance(pred_result, tuple):
        y_pred = pred_result[0]
    else:
        y_pred = pred_result

    accuracy = accuracy_score(y_test, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    report = classification_report(y_test, y_pred, target_names=label_names)

    return {
        "model_name": model_name,
        "task_name": task_name,
        "accuracy": accuracy,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "precision_macro": precision_m,
        "recall_macro": recall_m,
        "f1_macro": f1_m,
        "classification_report": report,
    }


def compare_models(results_list):
    header = (
        f"{'Model':<20} {'Task':<15} {'Accuracy':<10} "
        f"{'Prec(W)':<10} {'Recall(W)':<10} {'F1(W)':<10} "
        f"{'Prec(M)':<10} {'Recall(M)':<10} {'F1(M)':<10}"
    )
    lines = [header, "-" * len(header)]

    for r in results_list:
        line = (
            f"{r['model_name']:<20} {r['task_name']:<15} "
            f"{r['accuracy']:<10.4f} "
            f"{r['precision_weighted']:<10.4f} {r['recall_weighted']:<10.4f} "
            f"{r['f1_weighted']:<10.4f} "
            f"{r['precision_macro']:<10.4f} {r['recall_macro']:<10.4f} "
            f"{r['f1_macro']:<10.4f}"
        )
        lines.append(line)

    return "\n".join(lines)


def generate_evaluation_report(
    nb_cat_result, nb_sev_result, svm_cat_result, svm_sev_result,
    bert_cat_result=None, bert_sev_result=None, output_path=None
):
    results = [nb_cat_result, nb_sev_result, svm_cat_result, svm_sev_result]
    if bert_cat_result is not None:
        results.append(bert_cat_result)
    if bert_sev_result is not None:
        results.append(bert_sev_result)

    sections = []
    sections.append("# 缺陷分类系统评估报告\n")
    sections.append("生成时间: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    sections.append("对比模型: 朴素贝叶斯(NB) + 线性SVM + BERT(bert-base-chinese + 2层Transformer)\n\n")

    sections.append("## 模型对比\n")
    sections.append(
        "| 模型 | 任务 | 准确率 | 精确率(W) | 召回率(W) | F1(W) | 精确率(M) | 召回率(M) | F1(M) |\n"
    )
    sections.append("|------|------|--------|-----------|-----------|-------|-----------|-----------|-------|\n")
    for r in results:
        sections.append(
            "| %s | %s | %.4f "
            "| %.4f | %.4f "
            "| %.4f | %.4f "
            "| %.4f | %.4f |\n" % (
                r['model_name'], r['task_name'], r['accuracy'],
                r['precision_weighted'], r['recall_weighted'],
                r['f1_weighted'], r['precision_macro'],
                r['recall_macro'], r['f1_macro'],
            )
        )

    sections.append("\n## 详细分类报告\n")
    for r in results:
        sections.append("### %s - %s\n" % (r['model_name'], r['task_name']))
        sections.append("```\n%s\n```\n" % r['classification_report'])

    sections.append("\n## 总结与建议\n")
    cat_results = [r for r in results if r["task_name"] == "category"]
    sev_results = [r for r in results if r["task_name"] == "severity"]

    if cat_results:
        best_cat = max(cat_results, key=lambda x: x["f1_weighted"])
        sections.append(
            "- 分类任务最佳模型: **%s** (F1=%.4f)\n" % (best_cat['model_name'], best_cat['f1_weighted'])
        )
    if sev_results:
        best_sev = max(sev_results, key=lambda x: x["f1_weighted"])
        sections.append(
            "- 严重度任务最佳模型: **%s** (F1=%.4f)\n" % (best_sev['model_name'], best_sev['f1_weighted'])
        )

    if bert_cat_result is not None and bert_sev_result is not None:
        sections.append("\n### BERT vs 传统机器学习\n")
        nb_cat = [r for r in cat_results if r["model_name"] == "NB"][0] if any(r["model_name"] == "NB" for r in cat_results) else None
        svm_cat = [r for r in cat_results if r["model_name"] == "SVM"][0] if any(r["model_name"] == "SVM" for r in cat_results) else None
        if svm_cat is not None:
            cat_gain = bert_cat_result["f1_weighted"] - svm_cat["f1_weighted"]
            sections.append(
                "- 分类任务BERT相对SVM增益: %+.4f F1\n" % cat_gain
            )
        svm_sev = [r for r in sev_results if r["model_name"] == "SVM"][0] if any(r["model_name"] == "SVM" for r in sev_results) else None
        if svm_sev is not None:
            sev_gain = bert_sev_result["f1_weighted"] - svm_sev["f1_weighted"]
            sections.append(
                "- 严重度任务BERT相对SVM增益: %+.4f F1\n" % sev_gain
            )
        sections.append("\n")

    sections.append(
        "- 建议根据实际业务场景权衡精确率与召回率，必要时调整分类阈值。\n"
    )
    sections.append(
        "- 若模型表现差异较小，可考虑集成学习以提升整体稳定性。\n"
    )

    report = "".join(sections)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        return None

    return report


class EvaluationReport:
    def __init__(self):
        self._results = {}
        self._accuracy_history = {}

    def add_result(self, task, model_name, metrics_dict):
        key = "%s_%s" % (task, model_name)
        self._results[key] = {
            "task": task,
            "model_name": model_name,
            "metrics": metrics_dict,
        }

    def get_summary(self):
        if not self._results:
            return "暂无评估结果。"

        lines = ["评估结果汇总", "=" * 60]
        for key, entry in self._results.items():
            m = entry["metrics"]
            lines.append(
                "\n[%s] %s:\n"
                "  准确率: %.4f\n"
                "  F1(weighted): %.4f\n"
                "  F1(macro): %.4f" % (
                    entry['task'], entry['model_name'],
                    m.get('accuracy', 0),
                    m.get('f1_weighted', 0),
                    m.get('f1_macro', 0),
                )
            )
        return "\n".join(lines)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "results": self._results,
            "accuracy_history": self._accuracy_history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def track_accuracy_over_time(self, model_name, task, accuracy, version, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        key = "%s_%s" % (task, model_name)
        if key not in self._accuracy_history:
            self._accuracy_history[key] = []
        self._accuracy_history[key].append(
            {"version": version, "accuracy": accuracy, "timestamp": timestamp}
        )

    def get_accuracy_trend(self, model_name, task):
        key = "%s_%s" % (task, model_name)
        records = self._accuracy_history.get(key, [])
        return [(r["version"], r["accuracy"], r["timestamp"]) for r in records]
