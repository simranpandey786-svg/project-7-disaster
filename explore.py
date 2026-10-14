"""Part 1 - data exploration. Prints a summary and saves plots to reports/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from common import REPORTS_DIR, add_features, load_raw

REPORTS_DIR.mkdir(exist_ok=True)
df = add_features(load_raw())

print("Shape:", df.shape)
df.info()
print("\nMissing values (share of rows):")
print(df.isna().mean().round(3))
print("\nDuplicate tweet texts:", df["text"].duplicated().sum())
print("\nClass counts (1 = disaster):")
print(df["target"].value_counts().sort_index())

# 1. class distribution
counts = df["target"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(4.5, 3.5))
bars = ax.bar(["Not disaster (0)", "Disaster (1)"], counts.values, color=["#2e6f5e", "#b3261e"])
ax.bar_label(bars)
ax.set_ylabel("Tweets")
ax.set_title("Class distribution")
fig.tight_layout()
fig.savefig(REPORTS_DIR / "class_distribution.png", dpi=150)

# 2. tweet length by class
fig, ax = plt.subplots(figsize=(6, 3.5))
for label, name, color in [(0, "Not disaster", "#2e6f5e"), (1, "Disaster", "#b3261e")]:
    ax.hist(df.loc[df["target"] == label, "n_chars"], bins=30, alpha=0.6, label=name, color=color)
ax.set_xlabel("Characters per tweet")
ax.set_ylabel("Tweets")
ax.legend()
ax.set_title("Tweet length by class")
fig.tight_layout()
fig.savefig(REPORTS_DIR / "length_by_class.png", dpi=150)

# 3. most frequent words per class
vec = CountVectorizer(stop_words="english", min_df=3)
X = vec.fit_transform(df["clean_text"])
vocab = np.array(vec.get_feature_names_out())
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False)
for ax, (label, name, color) in zip(axes, [(0, "Not disaster", "#2e6f5e"), (1, "Disaster", "#b3261e")]):
    freq = np.asarray(X[(df["target"] == label).values].sum(axis=0)).ravel()
    top = freq.argsort()[::-1][:15][::-1]
    ax.barh(vocab[top], freq[top], color=color)
    ax.set_title(f"Top words: {name}")
fig.tight_layout()
fig.savefig(REPORTS_DIR / "top_words.png", dpi=150)

# 4. keywords (if the column exists): how often does each keyword mean a real disaster?
if "keyword" in df.columns:
    kw = df.dropna(subset=["keyword"]).copy()
    kw["keyword"] = kw["keyword"].str.replace("%20", " ", regex=False)
    stats = kw.groupby("keyword")["target"].agg(["mean", "count"])
    stats = stats[stats["count"] >= 5].sort_values("mean")
    pick = stats.head(12).index.tolist() + stats.tail(12).index.tolist()
    fig, ax = plt.subplots(figsize=(6, 7))
    sub = stats.loc[pick]
    ax.barh(sub.index, sub["mean"], color=["#2e6f5e" if v < 0.5 else "#b3261e" for v in sub["mean"]])
    ax.set_xlabel("Share of tweets that are real disasters")
    ax.set_title("Keywords that are most / least reliable")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "keyword_reliability.png", dpi=150)
    print("\nAmbiguous keywords (about 50/50):")
    print(stats[(stats["mean"] > 0.4) & (stats["mean"] < 0.6)].sort_values("count", ascending=False).head(10))

print(f"\nPlots saved to {REPORTS_DIR}")
