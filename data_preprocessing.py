# ============================================================
# data_preprocessing.py — Load, clean, and prepare the dataset
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV file."""
    df = pd.read_csv(config.DATA_PATH)
    print(f"[data] Loaded {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert G3 into binary PASS / FAIL label."""
    df = df.copy()
    df["pass"] = (df[config.TARGET_COL] >= config.PASS_THRESHOLD).astype(int)
    pass_rate = df["pass"].mean() * 100
    print(f"[data] Pass rate: {pass_rate:.1f}%  |  Fail rate: {100-pass_rate:.1f}%")
    return df


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that would leak the target at inference time."""
    drop_cols = config.LEAKAGE_COLS if not config.INCLUDE_GRADES else [config.TARGET_COL]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all nominal / binary categorical features."""
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"[data] After encoding: {df.shape[1]} features")
    return df


def split_and_scale(df: pd.DataFrame):
    """
    Split into train / test sets and apply StandardScaler.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    scaler                            : fitted StandardScaler
    feature_names                     : list[str]
    """
    y = df["pass"].values
    X = df.drop(columns=["pass"]).values
    feature_names = df.drop(columns=["pass"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[data] Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test, scaler, feature_names


def get_processed_data():
    """
    One-stop helper used by training scripts and the Streamlit app.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names, raw_df
    """
    df = load_raw_data()
    df = create_target(df)
    df = drop_leakage(df)
    df = encode_categoricals(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)
    return X_train, X_test, y_train, y_test, scaler, feature_names, df


if __name__ == "__main__":
    get_processed_data()
