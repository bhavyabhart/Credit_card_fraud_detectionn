# Credit Card Fraud Detection

Binary classification model that predicts whether a customer will default
(`bad_flag = 1`) using behavioural, transaction, and bureau data.

## Dataset

- Dataset folder: [Google Drive link](https://drive.google.com/drive/folders/1L3b09NTC3cK8Z1vtSk9vIsD-mzQDrQMq)

- **96,806** customers, **1,212** raw features across 4 feature families:
  - `onus` — 48 features
  - `transaction` — 664 features
  - `bureau` — 450 features
  - `bureau_enquiry` — 50 features
- Severely imbalanced target: **1.42% defaults** (1,372 positive / 95,434 negative)
- Split: 80/20 stratified train/test (`train_test_split(..., stratify=y)`)
  - Train: 77,444 rows · Test: 19,362 rows

## Pipeline

1. **Preprocessing (fit on train only, to avoid leakage)**
   Each feature family is imputed, scaled, and PCA-reduced independently,
   with the imputer/scaler/PCA fit strictly on `X_train` and only
   *applied* (not re-fit) to `X_test`:

   | Family | Raw features | PCA components | Variance explained |
   |---|---|---|---|
   | onus | 48 | 5 | 53.45% |
   | transaction | 664 | 5 | 13.32% |
   | bureau | 450 | 5 | 31.54% |
   | bureau_enquiry | 50 | 5 | 69.31% |

   The 4 family-level PCA outputs are concatenated into a final
   **20-dimensional** feature set (`X_train_pca`, `X_test_pca`).

2. **Class imbalance — SMOTE**
   SMOTE oversampling is applied to the PCA-reduced *training* data only
   (`X_train_pca`, never the test set), balancing 76,346 vs 1,098 defaults
   up to 76,346 vs 76,346.

3. **Models trained**
   - Logistic Regression (baseline, no resampling)
   - Logistic Regression + SMOTE
   - Random Forest + SMOTE
   - Gradient Boosting + SMOTE
   - Decision Tree + SMOTE
   - **Final model: Soft-Voting Ensemble** of (Logistic Regression + SMOTE,
     Random Forest + SMOTE, Gradient Boosting + SMOTE)

   No hyperparameter search (GridSearchCV/RandomizedSearchCV) is included
   in this run — models use manually chosen parameters. This is a known
   gap / natural next step (see Limitations).

4. **Threshold analysis**
   Precision/Recall/F1 swept across thresholds 0.01–0.50 on the ensemble's
   predicted probabilities; final reported metrics use threshold = 0.50.

## Results

Individual models (test set):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression – No SMOTE | 0.9857 | 0.0000 | 0.0000 | 0.0000 | 0.8080 | 0.0641 |
| Logistic Regression + SMOTE | 0.7347 | 0.0391 | 0.7518 | 0.0742 | 0.8079 | 0.0610 |
| Random Forest + SMOTE | 0.7885 | 0.0404 | 0.6131 | 0.0758 | 0.7904 | 0.0589 |
| Gradient Boosting + SMOTE | 0.6905 | 0.0345 | 0.7737 | 0.0661 | 0.7931 | 0.0558 |
| Decision Tree + SMOTE | 0.7023 | 0.0320 | 0.6861 | 0.0612 | 0.7592 | 0.0365 |

**Final model — Soft Voting Ensemble (threshold = 0.50):**

| Metric | Value |
|---|---|
| Accuracy | 0.7437 |
| Precision | 0.0384 |
| Recall | **0.7117** |
| F1 | 0.0729 |
| **ROC-AUC** | **0.8088** |
| PR-AUC | 0.0736 |

Confusion matrix (test set):

```
              Predicted 0   Predicted 1
Actual 0        14,204         4,884
Actual 1            79           195
```

Precision is intentionally low here — the threshold/ensemble is tuned to
catch defaulters (recall), which is the priority in a credit risk setting
where missing an actual default is costlier than a false alarm.



## How to Run

1. Open `CreditRisk.ipynb` in Google Colab.
2. Run the first cell to mount Google Drive when prompted.
3. Ensure `Dev_data_to_be_shared.csv` is available at:
   `/content/drive/MyDrive/Credit Card Fraud Detection/Dev_data_to_be_shared.csv`
   (or update the path in the data-loading cell).
4. Run all cells top to bottom (Runtime → Run All). No GPU required.

