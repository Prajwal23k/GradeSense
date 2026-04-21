# ============================================================
# visualize.py — EDA + model evaluation visualizations
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics       import confusion_matrix, roc_curve, auc as sk_auc
from sklearn.tree          import DecisionTreeClassifier, plot_tree

import config

warnings.filterwarnings("ignore")
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

PALETTE = "viridis"
sns.set_theme(style="whitegrid", palette=PALETTE)


# ─── 1. EDA ──────────────────────────────────────────────────────────────────

def plot_grade_distribution(df: pd.DataFrame):
    """Histogram of final grades + PASS/FAIL split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Grade histogram
    axes[0].hist(df["G3"], bins=range(0, 22), color="#4C72B0", edgecolor="white", rwidth=0.85)
    axes[0].axvline(config.PASS_THRESHOLD, color="#DD4444", linewidth=2, linestyle="--", label=f"Pass threshold ({config.PASS_THRESHOLD})")
    axes[0].set_title("Distribution of Final Grades (G3)")
    axes[0].set_xlabel("Grade")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # PASS / FAIL pie
    pass_counts = df["pass"].value_counts()
    axes[1].pie(
        pass_counts.values,
        labels=["PASS", "FAIL"] if pass_counts.index[0] == 1 else ["FAIL", "PASS"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("PASS vs FAIL Distribution")

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "01_grade_distribution.png")
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {path}")


def plot_feature_correlations(df: pd.DataFrame):
    """Heatmap of numeric feature correlations."""
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5, ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()

    path = os.path.join(config.OUTPUT_DIR, "02_correlation_heatmap.png")
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {path}")


def plot_study_vs_pass(df: pd.DataFrame):
    """Study time vs pass rate bar chart."""
    study_pass = df.groupby("studytime")["pass"].mean().reset_index()
    study_pass.columns = ["Study Time", "Pass Rate"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(study_pass["Study Time"], study_pass["Pass Rate"] * 100,
                  color=sns.color_palette(PALETTE, len(study_pass)), edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("Study Time (1=<2h … 4=>10h/week)")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Study Time vs Pass Rate")
    ax.set_ylim(0, 110)
    plt.tight_layout()

    path = os.path.join(config.OUTPUT_DIR, "03_studytime_vs_passrate.png")
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {path}")


def plot_failures_vs_pass(df: pd.DataFrame):
    """Number of past failures vs pass rate."""
    fp = df.groupby("failures")["pass"].mean().reset_index()
    fp.columns = ["Failures", "Pass Rate"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fp["Failures"], fp["Pass Rate"] * 100,
            marker="o", linewidth=2, markersize=8, color="#e74c3c")
    for _, row in fp.iterrows():
        ax.text(row["Failures"], row["Pass Rate"] * 100 + 1.5,
                f"{row['Pass Rate']*100:.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("Number of Past Failures")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Past Failures vs Pass Rate")
    ax.set_ylim(0, 110)
    plt.tight_layout()

    path = os.path.join(config.OUTPUT_DIR, "04_failures_vs_passrate.png")
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {path}")


# ─── 2. Model evaluation ─────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, model_name: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["FAIL", "PASS"],
        yticklabels=["FAIL", "PASS"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(config.OUTPUT_DIR, f"cm_{safe_name}.png")
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {path}")


def plot_roc_curves(roc_data: list):
    """
    roc_data: list of dicts → {"name": str, "y_test": array, "y_prob": array}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette(PALETTE, len(roc_data))

    for (rd, color) in zip(roc_data, colors):
        fpr, tpr, _ = roc_curve(rd["y_test"], rd["y_prob"])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{rd['name']} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(config.OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {path}")


def plot_model_comparison(results: list[dict]):
    """Bar chart comparing accuracy + ROC-AUC across all models."""
    df_r = pd.DataFrame(results)
    df_r = df_r.sort_values("accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_r))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_r["accuracy"], width,
                   label="Accuracy", color="#3498db", edgecolor="white")
    bars2 = ax.bar(x + width/2, df_r["roc_auc"],  width,
                   label="ROC-AUC",  color="#2ecc71", edgecolor="white")

    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df_r["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(config.OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {path}")


def plot_feature_importance(model, feature_names: list, model_name: str, top_n=20):
    """Works for tree-based models with feature_importances_ attribute."""
    if not hasattr(model, "feature_importances_"):
        print(f"[viz] {model_name} has no feature_importances_ — skipping.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_importances = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", top_n)
    bars = ax.barh(top_features[::-1], top_importances[::-1], color=colors[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(config.OUTPUT_DIR, f"feature_importance_{safe_name}.png")
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {path}")


def plot_ann_training_history(history):
    """Plot training vs validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("ANN — Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("ANN — Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "ann_training_history.png")
    plt.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {path}")
