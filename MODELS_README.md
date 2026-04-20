# Models Documentation

## Overview
This project uses a **Stacking Ensemble** approach combining multiple machine learning models to predict student pass/fail outcomes.

---

## Models Used

### 1. Random Forest
| Attribute | Details |
|-----------|---------|
| **File** | `train_ensemble.py` |
| **Type** | Base Learner (Level 0) |
| **Library** | `sklearn.ensemble.RandomForestClassifier` |
| **Configuration** | `n_estimators=200, max_depth=None, min_samples_split=5` |
| **Functionality** | Captures non-linear feature interactions using decision tree ensemble. Each tree votes, final prediction is majority vote. Handles mixed data types well. |

### 2. XGBoost
| Attribute | Details |
|-----------|---------|
| **File** | `train_ensemble.py` |
| **Type** | Base Learner (Level 0) |
| **Library** | `xgboost.XGBClassifier` |
| **Configuration** | `n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8` |
| **Functionality** | Gradient boosting model that builds trees sequentially, correcting previous errors. Handles class imbalance well. |

### 3. Logistic Regression
| Attribute | Details |
|-----------|---------|
| **File** | `train_ensemble.py` (used as both base learner and meta-learner) |
| **Type** | Base Learner (Level 0) & Meta-Learner (Level 1) |
| **Library** | `sklearn.linear_model.LogisticRegression` |
| **Configuration** | `C=1.0, solver='lbfgs', max_iter=1000` |
| **Functionality** | Linear model outputting probability. As base learner: captures linear patterns. As meta-learner: weights outputs from all base learners. |

### 4. Artificial Neural Network (ANN)
| Attribute | Details |
|-----------|---------|
| **File** | `train_ensemble.py` (training), `app.py` (inference) |
| **Type** | Base Learner (Level 0) |
| **Library** | `tensorflow.keras.Sequential` |
| **Architecture** | Input(128)→ReLU→BN→Dropout(0.3)→(64)→ReLU→BN→Dropout(0.2)→(32)→ReLU→Sigmoid(1) |
| **Functionality** | Deep learning model with 3 hidden layers. Learns complex patterns through non-linear transformations. |

### 5. Stacking Ensemble (Combined)
| Attribute | Details |
|-----------|---------|
| **File** | `train_ensemble.py`, `app.py` |
| **Type** | Final Model |
| **Architecture** | Level 0: RF + XGBoost + LR + ANN → Level 1: Logistic Regression |
| **Functionality** | Combines all 4 base learners' predictions using a meta-learner for final prediction. Reduces overfitting and improves generalization. |

---

## Model Files

| File | Description |
|------|------------|
| `models/ensemble.pkl` | Pickled base learners + meta-learner (sklearn models) |
| `models/ensemble_ann.keras` | Trained ANN model (TensorFlow/Keras) |
| `models/scaler.pkl` | Feature scaler (StandardScaler) |
| `models/feature_names.pkl` | Feature names list for alignment |

---

## Training Strategy

1. **5-Fold Cross-Validation**: Base learners trained with out-of-fold predictions
2. **Meta-learner Training**: Logistic Regression learns optimal weights from base predictions
3. **Final Retraining**: All models retrained on full dataset

---

## Prediction Flow

```
Input Features → Base Learners (RF, XGB, LR, ANN) → Probabilities
                     ↓
              Meta-Learner (Logistic Regression)
                     ↓
              Final PASS/FAIL Prediction
```