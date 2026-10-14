"""Shared paths, text cleaning and feature helpers used by every script."""
import html
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SEED = 42

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")

NUMERIC_FEATURES = ["n_chars", "n_words", "n_hashtags", "n_mentions", "n_urls"]


def clean_for_tfidf(text: str) -> str:
    """Aggressive cleaning for the bag-of-words model.

    Lower-cases, drops URLs / @mentions / digits / punctuation / odd symbols and
    keeps the word inside a hashtag (#earthquake -> earthquake).
    """
    text = html.unescape(str(text)).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_for_transformer(text: str) -> str:
    """Light cleaning for DistilBERT: keep punctuation and casing context."""
    text = html.unescape(str(text))
    text = URL_RE.sub("http", text)
    text = MENTION_RE.sub("@user", text)
    return re.sub(r"\s+", " ", text).strip()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the cleaned text and the simple numeric features used by TF-IDF models."""
    out = df.copy()
    raw = out["text"].astype(str)
    out["clean_text"] = raw.map(clean_for_tfidf)
    out["n_chars"] = raw.str.len()
    out["n_words"] = out["clean_text"].str.split().str.len()
    out["n_hashtags"] = raw.str.count(r"#\w+")
    out["n_mentions"] = raw.str.count(r"@\w+")
    out["n_urls"] = raw.str.count(r"https?://\S+|www\.\S+")
    return out


def load_raw(path=None) -> pd.DataFrame:
    path = Path(path) if path else DATA_DIR / "train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the disaster tweets CSV (columns: text, target) in data/ "
            "as train.csv - see README.md."
        )
    df = pd.read_csv(path)
    missing = {"text", "target"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")
    return df


def load_splits():
    train_path, test_path = DATA_DIR / "train_split.csv", DATA_DIR / "test_split.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run `python prepare_data.py` first.")
    return pd.read_csv(train_path), pd.read_csv(test_path)
