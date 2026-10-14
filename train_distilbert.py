"""Part 2 - fine-tune DistilBERT for binary tweet classification (plain PyTorch loop)."""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from common import MODELS_DIR, REPORTS_DIR, SEED, clean_for_transformer, load_splits

BASE_MODEL = "distilbert-base-uncased"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        gold.extend(batch["labels"].cpu().tolist())
    return f1_score(gold, preds), accuracy_score(gold, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=96)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print("Device:", device)

    train_df, _ = load_splits()  # the test split is never touched here
    train_df["clean"] = train_df["text"].map(clean_for_transformer)
    tr, val = train_test_split(train_df, test_size=0.1, stratify=train_df["target"], random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    collate = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        TweetDataset(tr["clean"], tr["target"], tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        TweetDataset(val["clean"], val["target"], tokenizer, args.max_length),
        batch_size=64, collate_fn=collate,
    )

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    out_dir = MODELS_DIR / "distilbert"
    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")
        val_f1, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train loss {running / len(train_loader):.4f}  val F1 {val_f1:.4f}  val acc {val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": running / len(train_loader), "val_f1": val_f1, "val_acc": val_acc})
        if val_f1 > best_f1:  # keep the best epoch, not the last one
            best_f1 = val_f1
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print("  saved new best model")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "distilbert_training_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest validation F1: {best_f1:.4f}. Model saved to {out_dir}")


if __name__ == "__main__":
    main()
