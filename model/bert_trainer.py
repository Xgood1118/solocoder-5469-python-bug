import os
import logging
import datetime

import joblib
import numpy as np

from config import BERT_MODEL_NAME, BERT_MAX_LENGTH, MODELS_DIR

logger = logging.getLogger(__name__)

BERT_AVAILABLE = None


def _check_bert_available():
    global BERT_AVAILABLE
    if BERT_AVAILABLE is not None:
        return BERT_AVAILABLE
    try:
        import torch
        import transformers
        _ = torch.__version__
        _ = transformers.__version__
        BERT_AVAILABLE = True
        logger.info("BERT/PyTorch is available")
    except Exception as e:
        BERT_AVAILABLE = False
        logger.warning("BERT not available: %s", e)
    return BERT_AVAILABLE


def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_label_map(labels):
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_label_map = {v: k for k, v in label_map.items()}
    return label_map, reverse_label_map


class _BertClassifierNN:
    @staticmethod
    def create(num_classes, num_hidden_layers=2, freeze_bert=True):
        import torch.nn as nn
        from transformers import BertModel

        class BertClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)

                if freeze_bert:
                    for param in self.bert.parameters():
                        param.requires_grad = False

                hidden_size = self.bert.config.hidden_size

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    batch_first=True,
                )
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=num_hidden_layers
                )

                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size // 2, num_classes),
                )

            def forward(self, input_ids, attention_mask, token_type_ids=None):
                bert_output = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                sequence_output = bert_output.last_hidden_state

                transformed = self.transformer_encoder(
                    sequence_output, src_key_padding_mask=(attention_mask == 0)
                )

                cls_output = transformed[:, 0, :]
                logits = self.classifier(cls_output)
                return logits

        return BertClassifier()


class BERTTrainer:
    def __init__(self, task, num_classes=None, num_hidden_layers=2):
        if not _check_bert_available():
            raise RuntimeError(
                "BERT is not available in this environment "
                "(PyTorch/transformers not installed or failed to load)"
            )
        if task not in ("category", "severity"):
            raise ValueError("task must be 'category' or 'severity', got '%s'" % task)

        self.task = task
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.model = None
        self.tokenizer = None
        self.label_map = None
        self.reverse_label_map = None
        self._device = None
        self._trained = False

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        return self.tokenizer

    def _encode_texts(self, texts, max_length=None):
        if max_length is None:
            max_length = BERT_MAX_LENGTH
        tokenizer = self._ensure_tokenizer()
        encoding = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoding

    def train(self, texts, labels, auxiliary_features=None,
              epochs=3, batch_size=16, lr=2e-5, max_length=BERT_MAX_LENGTH):
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        if self._device is None:
            self._device = _get_device()

        if self.num_classes is None:
            self.num_classes = len(set(labels))

        self.label_map, self.reverse_label_map = _build_label_map(labels)
        self._ensure_tokenizer()

        class DefectDataset(Dataset):
            def __init__(self_self):
                self_self.texts = texts
                self_self.labels = labels

            def __len__(self_self):
                return len(self_self.texts)

            def __getitem__(self_self, idx):
                text = self_self.texts[idx]
                label = self_self.labels[idx]

                tokenizer = self._ensure_tokenizer()
                encoding = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                label_idx = self.label_map.get(label, 0)

                return {
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                    "label": torch.tensor(label_idx, dtype=torch.long),
                }

        dataset = DefectDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = _BertClassifierNN.create(
            num_classes=self.num_classes,
            num_hidden_layers=self.num_hidden_layers,
            freeze_bert=True,
        )
        self.model.to(self._device)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels_batch = batch["label"].to(self._device)

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels_batch)

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info("BERT %s epoch %d/%d loss: %.4f", self.task, epoch + 1, epochs, avg_loss)

        self._trained = True
        return self

    def predict(self, texts, auxiliary_features=None):
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")

        import torch

        if self._device is None:
            self._device = _get_device()

        self.model.eval()

        if isinstance(texts, str):
            texts = [texts]

        predictions = []
        confidences = []

        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch_texts = texts[i:i + 16]
                encoding = self._encode_texts(batch_texts)

                input_ids = encoding["input_ids"].to(self._device)
                attention_mask = encoding["attention_mask"].to(self._device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                max_probs, pred_indices = torch.max(probs, dim=1)

                for idx, prob in zip(pred_indices.cpu().numpy(), max_probs.cpu().numpy()):
                    label = self.reverse_label_map.get(int(idx), str(int(idx)))
                    predictions.append(label)
                    confidences.append(float(prob))

        return np.array(predictions), np.array(confidences)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_data = {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "label_map": self.label_map,
            "reverse_label_map": self.reverse_label_map,
            "num_classes": self.num_classes,
            "num_hidden_layers": self.num_hidden_layers,
            "task": self.task,
            "tokenizer_name": BERT_MODEL_NAME,
            "max_length": BERT_MAX_LENGTH,
        }
        joblib.dump(save_data, path)

    @classmethod
    def load(cls, path):
        if not _check_bert_available():
            raise RuntimeError("BERT is not available in this environment")

        save_data = joblib.load(path)
        trainer = cls(
            task=save_data["task"],
            num_classes=save_data["num_classes"],
            num_hidden_layers=save_data.get("num_hidden_layers", 2),
        )
        trainer.label_map = save_data["label_map"]
        trainer.reverse_label_map = save_data["reverse_label_map"]
        trainer._device = _get_device()

        from transformers import BertTokenizer
        trainer.tokenizer = BertTokenizer.from_pretrained(
            save_data.get("tokenizer_name", BERT_MODEL_NAME)
        )
        trainer.model = _BertClassifierNN.create(
            num_classes=save_data["num_classes"],
            num_hidden_layers=save_data.get("num_hidden_layers", 2),
        )
        trainer.model.load_state_dict(save_data["model_state_dict"])
        trainer.model.to(trainer._device)
        trainer.model.eval()
        trainer._trained = True
        return trainer


def train_bert_models(train_data, version):
    if not _check_bert_available():
        logger.warning("BERT not available, skipping BERT training")
        return {}

    texts = [r["description"] for r in train_data]
    category_labels = [r["category"] for r in train_data]
    severity_labels = [r["severity"] for r in train_data]

    results = {}

    for task, labels in [("category", category_labels), ("severity", severity_labels)]:
        logger.info("Training BERT model for task: %s", task)
        trainer = BERTTrainer(task=task, num_hidden_layers=2)
        trainer.train(texts, labels, epochs=3, batch_size=16)

        model_path = os.path.join(MODELS_DIR, "%s_bert_%s.joblib" % (task, version))
        trainer.save(model_path)

        pred_labels, conf_scores = trainer.predict(texts)
        accuracy = float(np.mean(pred_labels == np.array(labels)))
        avg_confidence = float(np.mean(conf_scores))

        results["%s_bert" % task] = {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "model_path": model_path,
            "version": version,
            "trained_at": datetime.datetime.now().isoformat(),
            "n_samples": len(labels),
        }
        logger.info("BERT %s accuracy: %.4f", task, accuracy)

    return results
