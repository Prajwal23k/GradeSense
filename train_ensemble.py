# ============================================================
# train_ensemble.py
# Stacking Ensemble: RF + XGBoost + ANN → Logistic Regression
# ============================================================
"""
Architecture
------------
Level 0 (base learners):
  • Random Forest        — best at capturing non-linear feature interactions
  • XGBoost             — best gradient boosting, handles imbalance well
  • Logistic Regression — strong linear baseline, good calibration

Level 1 (meta-learner):
  • Logistic Regression — learns how to optimally weight base learner outputs

The ANN predictions are also included as a 4th base learner signal.

Training strategy:
  • Base learners trained with 5-fold out-of-fold (OOF) predictions
    so the meta-learner never trains on data the base learners saw.
  • Final base learners then retrained on full training set.
  • Meta-learner trained on OOF probability outputs.
"""

import os, pickle, warnings
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix, f1_score,
)
from sklearn.calibration     import CalibratedClassifierCV
from xgboost                 import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import config

os.makedirs(config.MODEL_DIR, exist_ok=True)


# ── ANN builder ──────────────────────────────────────────────────────────────
def build_ann(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_ann_fold(X_tr, y_tr, X_val, input_dim):
    """Train ANN on a fold, return probabilities on val set."""
    model = build_ann(input_dim)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    model.fit(
        X_tr, y_tr,
        validation_split=0.1,
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )
    return model.predict(X_val, verbose=0).flatten()


# ── Base learner configs ──────────────────────────────────────────────────────
def get_base_learners():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_split=5, n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, use_label_encoder=False,
            eval_metric="logloss", n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "logistic_regression": LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            random_state=config.RANDOM_STATE,
        ),
    }


# ── Main stacking trainer ─────────────────────────────────────────────────────
def train_stacking_ensemble(X_train, X_test, y_train, y_test):
    """
    Train stacked ensemble.
    Returns: final model dict, evaluation metrics dict, oof_preds array
    """
    n_train    = X_train.shape[0]
    input_dim  = X_train.shape[1]
    n_folds    = 5
    skf        = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=config.RANDOM_STATE)

    base_learners = get_base_learners()
    n_base        = len(base_learners) + 1   # +1 for ANN

    # ── Out-of-fold predictions (train meta-learner on these) ────────────────
    oof_train = np.zeros((n_train, n_base))
    test_preds = {name: np.zeros((X_test.shape[0], n_folds))
                  for name in list(base_learners.keys()) + ["ann"]}

    print("\n[Ensemble] Generating out-of-fold predictions …")
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  Fold {fold_idx+1}/{n_folds} …", end=" ", flush=True)
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        col = 0
        for name, model in base_learners.items():
            model.fit(X_tr, y_tr)
            oof_train[val_idx, col]       = model.predict_proba(X_val)[:, 1]
            test_preds[name][:, fold_idx] = model.predict_proba(X_test)[:, 1]
            col += 1

        # ANN
        ann_val_prob = train_ann_fold(X_tr, y_tr, X_val, input_dim)
        oof_train[val_idx, col] = ann_val_prob

        # For test predictions, train a fresh ANN on all tr data
        ann_full = build_ann(input_dim)
        ann_full.fit(
            X_tr, y_tr,
            epochs=60, batch_size=32, verbose=0,
            callbacks=[EarlyStopping(monitor="loss", patience=8,
                                      restore_best_weights=True)],
        )
        test_preds["ann"][:, fold_idx] = ann_full.predict(X_test, verbose=0).flatten()
        print("done")

    # Average test predictions across folds (standard stacking practice)
    test_meta = np.column_stack([
        test_preds[name].mean(axis=1)
        for name in list(base_learners.keys()) + ["ann"]
    ])

    # ── Train meta-learner on OOF predictions ────────────────────────────────
    print("\n[Ensemble] Training meta-learner (Logistic Regression) …")
    meta_learner = LogisticRegression(C=1.0, solver="lbfgs",
                                       random_state=config.RANDOM_STATE)
    meta_learner.fit(oof_train, y_train)

    # ── Retrain final base learners on FULL training set ─────────────────────
    print("[Ensemble] Retraining base learners on full training set …")
    final_base = get_base_learners()
    for name, model in final_base.items():
        model.fit(X_train, y_train)
        print(f"  {name} ✓")

    # Final ANN on full training data
    print("  ann …", end=" ", flush=True)
    final_ann = build_ann(input_dim)
    final_ann.fit(
        X_train, y_train,
        epochs=80, batch_size=32, verbose=0,
        callbacks=[
            EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-5),
        ],
    )
    print("✓")

    # ── Final evaluation ──────────────────────────────────────────────────────
    def get_test_meta_features(X):
        cols = []
        for name, model in final_base.items():
            cols.append(model.predict_proba(X)[:, 1])
        cols.append(final_ann.predict(X, verbose=0).flatten())
        return np.column_stack(cols)

    X_test_meta  = get_test_meta_features(X_test)
    y_prob_final = meta_learner.predict_proba(X_test_meta)[:, 1]
    y_pred_final = (y_prob_final >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_final)
    auc = roc_auc_score(y_test, y_prob_final)
    f1  = f1_score(y_test, y_pred_final)
    cm  = confusion_matrix(y_test, y_pred_final)

    print(f"\n{'='*55}")
    print("  STACKING ENSEMBLE — Final Performance")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(classification_report(y_test, y_pred_final,
                                 target_names=["FAIL", "PASS"]))
    print(f"  Confusion Matrix:\n{cm}")

    # ── Save ──────────────────────────────────────────────────────────────────
    ensemble = {
        "base_learners": final_base,
        "ann":           final_ann,
        "meta_learner":  meta_learner,
    }
    with open(os.path.join(config.MODEL_DIR, "ensemble.pkl"), "wb") as f:
        # Save sklearn parts; ANN saved separately
        pickle.dump({
            "base_learners": final_base,
            "meta_learner":  meta_learner,
        }, f)

    final_ann.save(os.path.join(config.MODEL_DIR, "ensemble_ann.keras"))
    print(f"\n  Saved → {config.MODEL_DIR}/ensemble.pkl")
    print(f"  Saved → {config.MODEL_DIR}/ensemble_ann.keras")

    metrics = {"accuracy": acc, "roc_auc": auc, "f1": f1}
    return ensemble, metrics, oof_train, y_prob_final


def predict_ensemble(ensemble, X):
    """
    Run inference through the stacking ensemble.
    Returns: (label, probability_of_pass)
    """
    base_learners = ensemble["base_learners"]
    ann           = ensemble["ann"]
    meta          = ensemble["meta_learner"]

    cols = []
    for model in base_learners.values():
        cols.append(model.predict_proba(X)[:, 1])
    cols.append(ann.predict(X, verbose=0).flatten())
    X_meta = np.column_stack(cols)

    prob   = meta.predict_proba(X_meta)[:, 1]
    label  = (prob >= 0.5).astype(int)
    return label, prob


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

    train_stacking_ensemble(X_tr, X_te, y_tr, y_te)
    print("\n[done] Ensemble training complete.")
