# ============================================================
# train_classical_models.py
# Logistic Regression, Decision Tree, Random Forest, KNN
# ============================================================

import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix,
)

import config
from data_preprocessing   import get_processed_data
from feature_engineering  import add_engineered_features

warnings.filterwarnings("ignore")
os.makedirs(config.MODEL_DIR, exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────────────────
def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}   |   ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["FAIL", "PASS"]))
    print(f"  Confusion Matrix:\n{cm}")

    return {"model": name, "accuracy": acc, "roc_auc": auc}


# ── Model definitions with hyper-parameter grids ──────────────────────────────
MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        {
            "max_depth":        [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "criterion":        ["gini", "entropy"],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [5, 10, None],
            "min_samples_split": [2, 5],
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan"],
        },
    ),
}


def train_classical(X_train, X_test, y_train, y_test) -> list[dict]:
    """Train all classical models with grid-search CV. Return results list."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    results = []

    for name, (base_model, param_grid) in MODELS.items():
        print(f"\n[training] Grid-searching {name} …")
        gs = GridSearchCV(
            base_model, param_grid,
            cv=cv, scoring="accuracy",
            n_jobs=-1, verbose=0,
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        print(f"  Best params : {gs.best_params_}")
        print(f"  CV accuracy : {gs.best_score_:.4f}")

        result = evaluate(name, best, X_test, y_test)
        results.append(result)

        # Persist model
        model_path = os.path.join(config.MODEL_DIR, f"{name.lower().replace(' ','_')}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best, f)
        print(f"  Saved → {model_path}")

    return results


if __name__ == "__main__":
    import pandas as pd
    raw = pd.read_csv(config.DATA_PATH)
    from feature_engineering import add_engineered_features
    enriched = add_engineered_features(raw)

    from data_preprocessing import create_target, drop_leakage, encode_categoricals, split_and_scale
    enriched = create_target(enriched)
    enriched = drop_leakage(enriched)
    enriched = encode_categoricals(enriched)
    X_tr, X_te, y_tr, y_te, scaler, feat_names = split_and_scale(enriched)

    results = train_classical(X_tr, X_te, y_tr, y_te)
    print("\n[done] Classical model training complete.")
