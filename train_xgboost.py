# ============================================================
# train_xgboost.py — XGBoost with RandomizedSearchCV tuning
# ============================================================

import os, pickle, warnings
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier

import config

warnings.filterwarnings("ignore")
os.makedirs(config.MODEL_DIR, exist_ok=True)


def train_xgboost(X_train, X_test, y_train, y_test) -> dict:
    """
    Train XGBoost with randomized hyper-parameter search.
    Returns evaluation dict.
    """
    cv = StratifiedKFold(n_splits=config.XGB_CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)

    base = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        base,
        param_distributions=config.XGB_PARAM_GRID,
        n_iter=20,
        cv=cv,
        scoring="accuracy",
        random_state=config.RANDOM_STATE,
        verbose=0,
        n_jobs=-1,
    )

    print("[XGBoost] Running RandomizedSearchCV …")
    search.fit(X_train, y_train)
    best = search.best_estimator_

    print(f"  Best params : {search.best_params_}")
    print(f"  CV accuracy : {search.best_score_:.4f}")

    # ── Evaluate ──────────────────────────────────────────────
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print("  XGBoost (tuned)")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}   |   ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["FAIL", "PASS"]))
    print(f"  Confusion Matrix:\n{cm}")

    # ── Save ──────────────────────────────────────────────────
    model_path = os.path.join(config.MODEL_DIR, "xgboost.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best, f)
    print(f"  Saved → {model_path}")

    return {"model": "XGBoost", "accuracy": acc, "roc_auc": auc}


if __name__ == "__main__":
    import pandas as pd
    from feature_engineering import add_engineered_features
    from data_preprocessing  import (create_target, drop_leakage,
                                      encode_categoricals, split_and_scale)

    raw = pd.read_csv(config.DATA_PATH)
    df  = add_engineered_features(raw)
    df  = create_target(df)
    df  = drop_leakage(df)
    df  = encode_categoricals(df)
    X_tr, X_te, y_tr, y_te, scaler, _ = split_and_scale(df)

    train_xgboost(X_tr, X_te, y_tr, y_te)
    print("[done] XGBoost training complete.")
