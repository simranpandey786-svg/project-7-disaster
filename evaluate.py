"""Part 3 - evaluate every trained model on the held-out test split and compare them."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from common import REPORTS_DIR, load_splits
from predictors import available_predictors

ap = argparse.ArgumentParser()
ap.add_argument("--no-train-metrics", action="store_true", help="skip train-set scoring (faster)")
args = ap.parse_args()

REPORTS_DIR.mkdir(exist_ok=True)
train, test = load_splits()
predictors = available_predictors()
if not predictors:
    raise SystemExit("No trained models found in models/. Run train_tfidf.py / train_distilbert.py first.")

rows, test_probs = [], {}
for name, model in predictors.items():
    print("Scoring", name)
    probs = model.predict_proba(test["text"].tolist())
    pred = (probs >= 0.5).astype(int)
    y = test["target"]
    row = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, probs),
    }
    if not args.no_train_metrics:  # a big train/test gap signals overfitting
        train_pred = (model.predict_proba(train["text"].tolist()) >= 0.5).astype(int)
        row["train_f1"] = f1_score(train["target"], train_pred)
        row["f1_gap"] = row["train_f1"] - row["f1"]
    rows.append(row)
    test_probs[name] = (probs, pred)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["Not disaster", "Disaster"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"confusion_{name.replace(' ', '_').replace('+', '')}.png", dpi=150)
    plt.close(fig)

    errors = test.assign(probability=probs, predicted=pred)
    errors[errors["predicted"] != errors["target"]].to_csv(
        REPORTS_DIR / f"errors_{name.replace(' ', '_').replace('+', '')}.csv", index=False
    )

metrics = pd.DataFrame(rows).set_index("model").round(4)
metrics.to_csv(REPORTS_DIR / "metrics.csv")
print("\n", metrics.to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for name, (probs, _) in test_probs.items():
    RocCurveDisplay.from_predictions(test["target"], probs, name=name, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(test["target"], probs, name=name, ax=axes[1])
axes[0].set_title("ROC curves")
axes[1].set_title("Precision-recall curves")
fig.tight_layout()
fig.savefig(REPORTS_DIR / "roc_pr_curves.png", dpi=150)
print(f"\nMetrics, curves, confusion matrices and misclassified tweets saved to {REPORTS_DIR}")
