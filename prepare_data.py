"""Part 1 - data preparation: dedupe, clean, stratified train/test split."""
from sklearn.model_selection import train_test_split

from common import DATA_DIR, SEED, add_features, load_raw

df = load_raw()
df = df.dropna(subset=["text", "target"])
df["target"] = df["target"].astype(int)  # 0 = not disaster, 1 = disaster

# Exact duplicate tweets (some carry conflicting labels) would leak between train and
# test and inflate the scores, so keep one copy of each text.
before = len(df)
df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
print(f"Dropped {before - len(df)} duplicate tweets; {len(df)} remain")

train, test = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=SEED)
train.to_csv(DATA_DIR / "train_split.csv", index=False)
test.to_csv(DATA_DIR / "test_split.csv", index=False)

print(f"Train: {len(train)}  Test: {len(test)}")
print("Disaster share  train: %.3f  test: %.3f" % (train["target"].mean(), test["target"].mean()))

demo = add_features(train.head(3))
for raw, clean in zip(demo["text"], demo["clean_text"]):
    print("\nRAW   :", raw)
    print("CLEAN :", clean)
    print("TOKENS:", clean.split())
