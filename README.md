# Credit Risk Assessment

Predicting whether a borrower will default on a loan, using applicant and loan details. The project compares a Logistic Regression baseline with a tuned XGBoost model. It also covers class imbalance, probability calibration and model explainability with SHAP.

> **Status: in progress.** Modelling is complete and the final model is saved. A FastAPI prediction API (`main.py`) is working. The error analysis (section 15) and a web front end are underway.

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
16. **Save the final model**: find the best threshold for the calibrated model, then save the model and threshold with `joblib`

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
| XGBoost (tuned) | 0.92 | 0.83 | 0.81 | 0.82 |
| XGBoost (tuned + calibrated, threshold 0.71) | 0.94 | 0.97 | 0.74 | 0.84 |

The tuned XGBoost reached a cross-validated average precision of 0.90. Best parameters (`random_state=42`):
`n_estimators=435`, `max_depth=6`, `learning_rate≈0.063`, `subsample≈0.95`, `colsample_bytree≈0.85`, `min_child_weight=4`, `gamma≈1.48`.

The final model is the tuned XGBoost after sigmoid calibration. Its decision threshold of 0.71 is the one that maximizes F1 on the precision-recall curve. Compared with the default 0.5 threshold, it gives much higher precision but lower recall.

At the default threshold, the tuned model makes 221 false positives and 264 false negatives on the test set. Missed defaults are more costly to a lender, so this is the focus of the error analysis.

## Using the saved model

Running the notebook creates `credit_risk_model.pkl` (the calibrated pipeline, including preprocessing) and `best_threshold.pkl`. These files are not committed to the repo.

```python
import joblib

model = joblib.load("credit_risk_model.pkl")
threshold = joblib.load("best_threshold.pkl")

# new_applicants: DataFrame with the same feature columns as the dataset (without loan_status)
default_prob = model.predict_proba(new_applicants)[:, 1]
predicted_default = (default_prob > threshold).astype(int)
```

## Prediction API

`main.py` serves the saved model as a REST API built with FastAPI.

- On startup, it loads `credit_risk_model.pkl` and `best_threshold.pkl`.
- `POST /predict` takes one loan application and returns the probability of default. It labels the applicant **High Risk** if the probability is at or above the saved threshold, and **Low Risk** otherwise.
- The empty `static/` folder will hold the web front end (`index.html`), which will be served at `/`.

**Run it**. Run the notebook first to create the `.pkl` files, then start the server from the project folder:

```bash
uvicorn main:app --reload
```

Interactive API docs are at http://127.0.0.1:8000/docs.

**Example request**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"person_age": 22, "person_income": 59000, "person_home_ownership": "RENT",
       "person_emp_length": 5, "loan_intent": "PERSONAL", "loan_grade": "D",
       "loan_amnt": 35000, "loan_int_rate": 16.02, "loan_percent_income": 0.59,
       "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 3}'
```

**Example response**

```json
{
  "default_probability": 0.958,
  "default_prediction": 1,
  "threshold": 0.711,
  "Result": "High Risk"
}
```

## Next steps

- Build the web front end in `static/` (form for entering applicant details)
- Add input validation to the API (allowed category values, positive numbers)
- Finish the false positive / false negative analysis
- Pick the classification threshold on a validation split, not the test set
- Use the tuned model consistently for calibration and SHAP
- Summarize key risk drivers and business takeaways

## Running the notebook

```bash
git clone https://github.com/aishu-0610/Credit-Risk-Assessment.git
cd Credit-Risk-Assessment
pip install -r requirements.txt
jupyter notebook Credit_Risk.ipynb
```

Developed with Python 3.13.

## Project structure

```
├── Credit_Risk.ipynb         # Main notebook: EDA, modelling, evaluation, SHAP
├── credit_risk_dataset.csv   # Dataset
├── main.py                   # FastAPI prediction API
├── static/                   # Web front end (in progress)
├── requirements.txt          # Python dependencies
└── README.md
```
