# Credit Risk Assessment

Predicting whether a borrower will default on a loan, using applicant and loan details. The project compares a Logistic Regression baseline with a tuned XGBoost model. It also covers class imbalance, probability calibration and model explainability with SHAP.

> **Status: in progress.** Sections 1–14 are complete. Section 15 (error analysis of false positives and false negatives) is underway.

## Dataset

`credit_risk_dataset.csv` is the public Credit Risk Dataset from Kaggle. It has 32,581 loans and 12 columns.

| Column | Description |
|---|---|
| `person_age` | Applicant age |
| `person_income` | Annual income |
| `person_home_ownership` | RENT, OWN, MORTGAGE, OTHER |
| `person_emp_length` | Employment length (years) |
| `loan_intent` | Purpose of the loan (EDUCATION, MEDICAL, PERSONAL, ...) |
| `loan_grade` | Loan grade (A–G) |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate |
| `loan_percent_income` | Loan amount as a fraction of income |
| `cb_person_default_on_file` | Previous default on record (Y/N) |
| `cb_person_cred_hist_length` | Credit history length (years) |
| **`loan_status`** | **Target**: 1 = default, 0 = no default |

The target is imbalanced: 25,473 non-defaults vs 7,108 defaults (about 22% positive). `person_emp_length` has 895 missing values and `loan_int_rate` has 3,116.

## Workflow

All work is in [`Credit_Risk.ipynb`](Credit_Risk.ipynb):

1. **Load the data**
2. **Exploratory data analysis**: missing values, target balance, outliers (box plots), correlations
3. **Data validation and outlier handling**: drop duplicates, unrealistic ages (outside 18–100), and employment lengths longer than the applicant's age or above 60 years
4. **Features, target and train/test split**: 80/20 stratified split
5. **Class imbalance**: `class_weight="balanced"` for Logistic Regression and `scale_pos_weight` (≈3.63) for XGBoost
6. **Preprocessing pipelines**: `ColumnTransformer` with median imputation and scaling for numeric columns (scaling only for LR), and constant imputation plus one-hot encoding for categorical columns
7. **Evaluation helper**: accuracy, precision, recall, F1, confusion matrix, precision-recall curve
8. **Stratified 5-fold cross-validation** for both models
9. **Baseline model**: Logistic Regression
10. **XGBoost and hyperparameter tuning**: `RandomizedSearchCV` (150 candidates × 5 folds), optimizing average precision
11. **Side-by-side model comparison**
12. **Classification threshold optimization**: choose the threshold that maximizes F1 on the precision-recall curve
13. **Probability calibration**: Platt (sigmoid) scaling with `CalibratedClassifierCV` and calibration curves
14. **SHAP interpretation**: global summary plot and a local waterfall plot for a single applicant
15. **Error analysis** *(in progress)*: looking at false positives and false negatives from the tuned model

## Results so far

**5-fold cross-validation (training set)**

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.871 | 0.812 | 0.545 | 0.778 | 0.641 |
| XGBoost | 0.947 | 0.918 | 0.819 | 0.796 | 0.807 |

**Held-out test set (6,305 loans)**, metrics for the default class:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.82 | 0.56 | 0.79 | 0.65 |
| XGBoost (tuned) | 0.92 | 0.83 | 0.80 | 0.82 |

The tuned XGBoost reached a cross-validated average precision of 0.90. Best parameters:
`n_estimators=506`, `max_depth=5`, `learning_rate≈0.079`, `subsample≈0.89`, `colsample_bytree≈0.82`, `min_child_weight=3`, `gamma≈0.70`.

On the test set, the tuned model makes 224 false positives and 270 false negatives. Missed defaults are more costly to a lender, so this is the focus of the error analysis.

## Next steps

- Finish the false positive / false negative analysis
- Pick the classification threshold on a validation split, not the test set
- Use the tuned model consistently for calibration and SHAP
- Summarize key risk drivers and business takeaways

## Running the notebook

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook Credit_Risk.ipynb
```

Developed with Python 3.13.

## Project structure

```
├── Credit_Risk.ipynb         # Main notebook: EDA, modelling, evaluation, SHAP
├── credit_risk_dataset.csv   # Dataset
├── requirements.txt          # Python dependencies
└── README.md
```
