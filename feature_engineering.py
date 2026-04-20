# ============================================================
# feature_engineering.py — Domain-driven feature engineering
# ============================================================
"""
We create five new composite features before encoding / scaling:

  alcohol_total   — combined weekday + weekend alcohol index
  study_fail_ratio— ratio of study time to number of past failures
  social_score    — average of going-out + free-time (social exposure)
  parental_edu    — mean parental education level
  support_score   — count of active support services (school / family / paid)

All transformations operate on the *raw* DataFrame (before one-hot
encoding) so the feature names stay human-readable.
"""

import pandas as pd
import numpy as np


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the raw DataFrame in-place (returns copy).

    Parameters
    ----------
    df : raw student DataFrame (columns as in student-mat.csv)

    Returns
    -------
    df : DataFrame with extra feature columns appended
    """
    df = df.copy()

    # 1. Total alcohol consumption (weekday + weekend)
    df["alcohol_total"] = df["Dalc"] + df["Walc"]

    # 2. Study time per failure (higher → student studies more despite failures)
    #    Add 1 to denominator to avoid division by zero
    df["study_fail_ratio"] = df["studytime"] / (df["failures"] + 1)

    # 3. Social exposure score
    df["social_score"] = (df["goout"] + df["freetime"]) / 2.0

    # 4. Mean parental education
    df["parental_edu"] = (df["Medu"] + df["Fedu"]) / 2.0

    # 5. Support score: count of yes/no support columns
    support_cols = ["schoolsup", "famsup", "paid"]
    existing = [c for c in support_cols if c in df.columns]
    for col in existing:
        df[col + "_bin"] = (df[col] == "yes").astype(int)
    bin_cols = [c + "_bin" for c in existing]
    df["support_score"] = df[bin_cols].sum(axis=1)
    df.drop(columns=bin_cols, inplace=True)

    # 6. High-risk flag: failures > 1 AND absences > median
    median_absences = df["absences"].median()
    df["high_risk"] = (
        (df["failures"] > 1) & (df["absences"] > median_absences)
    ).astype(int)

    print(f"[feature_eng] Added 6 engineered features → {df.shape[1]} total cols")
    return df


if __name__ == "__main__":
    import pandas as pd, config
    raw = pd.read_csv(config.DATA_PATH)
    enriched = add_engineered_features(raw)
    print(enriched[["alcohol_total", "study_fail_ratio",
                     "social_score", "parental_edu",
                     "support_score", "high_risk"]].describe())
