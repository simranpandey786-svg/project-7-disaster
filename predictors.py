"""One interface for every trained model: predict_proba(list_of_texts) -> P(disaster)."""
import joblib
import numpy as np
import pandas as pd

from common import MODELS_DIR, add_features, clean_for_transformer


class TfidfPredictor:
    def __init__(self, path):
        self.pipeline = joblib.load(path)

    def predict_proba(self, texts):
        frame = add_features(pd.DataFrame({"text": list(texts)}))
        return self.pipeline.predict_proba(frame)[:, 1]


class DistilBertPredictor:
    def __init__(self, path, max_length=96, batch_size=32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.max_length, self.batch_size = max_length, batch_size
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()

    def predict_proba(self, texts):
        texts = [clean_for_transformer(t) for t in texts]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(
                    texts[i : i + self.batch_size],
                    padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[:, 1]
                out.append(probs.cpu().numpy())
        return np.concatenate(out) if out else np.array([])


def available_predictors():
    """Load every trained model found in models/. DistilBERT comes first when present."""
    found = {}
    bert_dir = MODELS_DIR / "distilbert"
    if (bert_dir / "config.json").exists():
        try:
            found["DistilBERT"] = DistilBertPredictor(bert_dir)
        except ImportError as exc:
            print("Skipping DistilBERT (install torch + transformers):", exc)
    for key, label in [("logreg", "TF-IDF + Logistic Regression"), ("random_forest", "TF-IDF + Random Forest")]:
        path = MODELS_DIR / f"tfidf_{key}.joblib"
        if path.exists():
            found[label] = TfidfPredictor(path)
    return found
