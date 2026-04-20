# ============================================================
# train_all.py — Master training pipeline
# ============================================================

import os, pickle, warnings
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import config
from feature_engineering     import add_engineered_features
from data_preprocessing      import (
    load_raw_data, create_target, drop_leakage,
    encode_categoricals, split_and_scale,
)
from train_classical_models  import train_classical
from train_xgboost           import train_xgboost
from train_ann               import train_ann
from train_ensemble          import train_stacking_ensemble, predict_ensemble
from visualize import (
    plot_grade_distribution, plot_feature_correlations,
    plot_study_vs_pass, plot_failures_vs_pass,
    plot_confusion_matrix, plot_roc_curves,
    plot_model_comparison, plot_feature_importance,
    plot_ann_training_history,
)

os.makedirs(config.MODEL_DIR,  exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def main():
    print("\n" + "="*60)
    print("  STUDENT PERFORMANCE PREDICTOR — Full Training Pipeline")
    print("="*60)

    # 1. Load & feature engineering
    print("\n[1/6] Loading data and engineering features ...")
    raw = load_raw_data()

    raw_eda = create_target(raw.copy())
    plot_grade_distribution(raw_eda)
    plot_feature_correlations(raw_eda)
    plot_study_vs_pass(raw_eda)
    plot_failures_vs_pass(raw_eda)

    df = add_engineered_features(raw)
    df = create_target(df)
    df = drop_leakage(df)
    df = encode_categoricals(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    with open(os.path.join(config.MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(config.MODEL_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    # 2. Individual models
    print("\n[2/6] Training classical models ...")
    classical_results = train_classical(X_train, X_test, y_train, y_test)

    print("\n[3/6] Training XGBoost ...")
    xgb_result = train_xgboost(X_train, X_test, y_train, y_test)

    print("\n[4/6] Training ANN ...")
    ann_result = train_ann(X_train, X_test, y_train, y_test)

    individual_results = classical_results + [xgb_result] + [
        {k: v for k, v in ann_result.items() if k != "history"}
    ]

    # 3. Stacking Ensemble (production model)
    print("\n[5/6] Building Stacking Ensemble (RF + XGBoost + LR + ANN) ...")
    ensemble, ensemble_metrics, oof_preds, ensemble_test_probs = \
        train_stacking_ensemble(X_train, X_test, y_train, y_test)

    ensemble_result = {
        "model":    "Stacking Ensemble",
        "accuracy": ensemble_metrics["accuracy"],
        "roc_auc":  ensemble_metrics["roc_auc"],
    }
    all_results = individual_results + [ensemble_result]

    # 4. Visualizations
    print("\n[6/6] Generating visualizations ...")
    plot_model_comparison(all_results)

    _, ens_probs = predict_ensemble(ensemble, X_test)
    ens_preds    = (ens_probs >= 0.5).astype(int)
    plot_confusion_matrix(y_test, ens_preds, "Stacking Ensemble")

    import glob
    for model_file in glob.glob(os.path.join(config.MODEL_DIR, "*.pkl")):
        mname = os.path.basename(model_file).replace(".pkl", "")
        if mname in ("scaler", "feature_names", "ensemble"):
            continue
        with open(model_file, "rb") as f:
            m = pickle.load(f)
        if not hasattr(m, "predict"):
            continue
        friendly = mname.replace("_", " ").title()
        plot_confusion_matrix(y_test, m.predict(X_test), friendly)
        plot_feature_importance(m, feature_names, friendly)

    roc_data = []
    for model_file in glob.glob(os.path.join(config.MODEL_DIR, "*.pkl")):
        mname = os.path.basename(model_file).replace(".pkl", "")
        if mname in ("scaler", "feature_names", "ensemble"):
            continue
        with open(model_file, "rb") as f:
            m = pickle.load(f)
        if hasattr(m, "predict_proba"):
            roc_data.append({
                "name":   mname.replace("_", " ").title(),
                "y_test": y_test,
                "y_prob": m.predict_proba(X_test)[:, 1],
            })
    roc_data.append({"name": "Stacking Ensemble", "y_test": y_test, "y_prob": ens_probs})
    plot_roc_curves(roc_data)

    if "history" in ann_result:
        plot_ann_training_history(ann_result["history"])

    # Summary
    df_summary = pd.DataFrame(all_results).sort_values("accuracy", ascending=False)
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(df_summary[["model", "accuracy", "roc_auc"]].to_string(index=False))

    best = df_summary.iloc[0]
    print(f"\nBest Model : {best['model']}")
    print(f"Accuracy   : {best['accuracy']:.4f}")
    print(f"ROC-AUC    : {best['roc_auc']:.4f}")

    df_summary.to_csv(os.path.join(config.OUTPUT_DIR, "model_results.csv"), index=False)
    print(f"\n[done] Production model: Stacking Ensemble")
    print(f"[done] Launch: streamlit run app.py")


if __name__ == "__main__":
    main()
