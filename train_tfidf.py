"""Part 2 - TF-IDF feature engineering + classical models with cross-validated grid search."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS_DIR, NUMERIC_FEATURES, REPORTS_DIR, SEED, add_features, load_splits


def make_pipeline(clf):
    features = ColumnTransformer(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True), "clean_text"),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("features", features), ("clf", clf)])


CANDIDATES = {
    "logreg": (
        LogisticRegression(max_iter=2000),
        {"clf__C": [0.5, 1, 2, 5, 10], "features__tfidf__ngram_range": [(1, 1), (1, 2)]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 50], "clf__min_samples_leaf": [1, 2]},
    ),
}

if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    train, _ = load_splits()
    train = add_features(train)
    X, y = train[["clean_text"] + NUMERIC_FEATURES], train["target"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    summary = []
    for name, (clf, grid) in CANDIDATES.items():
        print(f"\n=== {name}: grid search with 5-fold CV ===")
        search = GridSearchCV(make_pipeline(clf), grid, cv=cv, scoring="f1", n_jobs=-1, refit=True, verbose=1)
        search.fit(X, y)
        print("Best params:", search.best_params_)
        print("Best CV F1 : %.4f" % search.best_score_)
        joblib.dump(search.best_estimator_, MODELS_DIR / f"tfidf_{name}.joblib")
        pd.DataFrame(search.cv_results_).to_csv(REPORTS_DIR / f"tfidf_{name}_cv_results.csv", index=False)
        summary.append({"model": name, "cv_f1": search.best_score_, "best_params": str(search.best_params_)})

    pd.DataFrame(summary).to_csv(REPORTS_DIR / "tfidf_cv_summary.csv", index=False)
    print("\nSaved models to", MODELS_DIR)
