# 🎓 Student Performance Predictor

A complete end-to-end machine learning project that predicts whether a student
will **PASS** or **FAIL** their math course based on demographic, social, and
academic features from the UCI Student Performance dataset.

---

## 📁 Project Structure

```
student_performance_predictor/
│
├── data/
│   └── student-mat.csv          # Raw dataset
│
├── models/                      # Saved models (created after training)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── xgboost.pkl
│   ├── ann_model.keras
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── outputs/                     # Charts and results (created after training)
│   ├── 01_grade_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_studytime_vs_passrate.png
│   ├── 04_failures_vs_passrate.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── ann_training_history.png
│   └── model_results.csv
│
├── config.py                    # Central configuration (paths, hyperparams)
├── data_preprocessing.py        # Load, clean, encode, split, scale
├── feature_engineering.py       # Domain-driven feature creation
├── train_classical_models.py    # LR, Decision Tree, Random Forest, KNN
├── train_xgboost.py             # XGBoost with RandomizedSearchCV
├── train_ann.py                 # ANN (TensorFlow / Keras)
├── train_all.py                 # Master script — runs everything
├── visualize.py                 # Matplotlib + Seaborn plots
├── app.py                       # Streamlit prediction UI
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train all models

```bash
python train_all.py
```

This single command will:
- Load and preprocess the dataset
- Apply feature engineering (6 new features)
- Train **5 models** with hyperparameter tuning:
  - Logistic Regression (GridSearchCV)
  - Decision Tree (GridSearchCV)
  - Random Forest (GridSearchCV)
  - KNN (GridSearchCV)
  - XGBoost (RandomizedSearchCV)
- Train an **Artificial Neural Network** (Keras)
- Generate all EDA and evaluation visualizations
- Print a ranked results summary
- Save all models to `models/` and all plots to `outputs/`

### 3. Launch the web app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🧠 Models & Techniques

| Model | Tuning |
|---|---|
| Logistic Regression | GridSearchCV (C, solver) |
| Decision Tree | GridSearchCV (depth, split, criterion) |
| Random Forest | GridSearchCV (estimators, depth, split) |
| KNN | GridSearchCV (k, weights, metric) |
| XGBoost | RandomizedSearchCV (depth, LR, subsample …) |
| ANN (Keras) | EarlyStopping + ReduceLROnPlateau |

---

## 🔧 Feature Engineering

Six engineered features are created on top of the raw dataset:

| Feature | Description |
|---|---|
| `alcohol_total` | Weekday + weekend alcohol combined |
| `study_fail_ratio` | Study time ÷ (failures + 1) |
| `social_score` | Average of go-out + free-time |
| `parental_edu` | Mean of mother's + father's education |
| `support_score` | Count of active support services |
| `high_risk` | Flag: failures > 1 AND absences > median |

---

## 🎯 Target Definition

```
G3 >= 10  →  PASS  (label 1)
G3  < 10  →  FAIL  (label 0)
```

> **Note:** By default, intermediate grades G1 and G2 are excluded to
> simulate a realistic early-prediction scenario. Set `INCLUDE_GRADES = True`
> in `config.py` to include them (gives near-perfect accuracy but
> represents data leakage in real deployment).

---

## ⚙️ Configuration

All tunable settings live in `config.py`:

```python
PASS_THRESHOLD  = 10          # Grade threshold for PASS/FAIL
INCLUDE_GRADES  = False       # Include G1/G2 (leakage warning)
TEST_SIZE       = 0.20        # Train/test split ratio
RANDOM_STATE    = 42          # Reproducibility seed
ANN_EPOCHS      = 80          # Max ANN training epochs
ANN_BATCH_SIZE  = 32          # ANN batch size
```

---

## 📊 Output Visualizations

After running `train_all.py`, the `outputs/` folder contains:

- **Grade Distribution** — histogram + PASS/FAIL pie chart
- **Correlation Heatmap** — feature relationships
- **Study Time vs Pass Rate** — bar chart
- **Failures vs Pass Rate** — line chart
- **Model Comparison** — grouped bar (Accuracy + AUC)
- **ROC Curves** — all models on one plot
- **Confusion Matrices** — per model
- **Feature Importances** — for tree-based models
- **ANN Training History** — accuracy + loss curves

---

## 📦 Dataset

**UCI Student Performance Dataset**  
Source: https://archive.ics.uci.edu/ml/datasets/Student+Performance  
395 students · 33 features · Portuguese secondary school (Math course)

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Scikit-learn** — classical ML
- **XGBoost** — gradient boosting
- **TensorFlow / Keras** — deep learning
- **Streamlit** — web UI
- **Matplotlib + Seaborn** — visualizations
- **Pandas + NumPy** — data wrangling
