# ============================================================
# train_ann.py — Artificial Neural Network (TensorFlow / Keras)
# ============================================================

import os, warnings
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF info logs
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix,
)

import config

os.makedirs(config.MODEL_DIR, exist_ok=True)


def build_ann(input_dim: int) -> Sequential:
    """
    Architecture:
        Input → Dense(128) → BN → Dropout(0.3)
              → Dense(64)  → BN → Dropout(0.2)
              → Dense(32)
              → Dense(1, sigmoid)
    """
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


def train_ann(X_train, X_test, y_train, y_test) -> dict:
    """Train ANN and save. Returns evaluation dict."""
    input_dim = X_train.shape[1]
    model = build_ann(input_dim)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    print("\n[ANN] Training …")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=config.ANN_EPOCHS,
        batch_size=config.ANN_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print("  Artificial Neural Network")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}   |   ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["FAIL", "PASS"]))
    print(f"  Confusion Matrix:\n{cm}")

    # ── Save ──────────────────────────────────────────────────
    model_path = os.path.join(config.MODEL_DIR, "ann_model.keras")
    model.save(model_path)
    print(f"  Saved → {model_path}")

    return {"model": "ANN", "accuracy": acc, "roc_auc": auc, "history": history}


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

    train_ann(X_tr, X_te, y_tr, y_te)
    print("[done] ANN training complete.")
