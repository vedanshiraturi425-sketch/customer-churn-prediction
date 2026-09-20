# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their service, using the Telco Customer Churn dataset (7,043 records, 21 features).

## What I did
- Cleaned the data (fixed `TotalCharges` stored as text, dropped 11 blank rows)
- EDA on churn drivers: contract type, tenure, monthly charges
- One-hot encoded categorical features (20 to 30 columns) and standardised them
- Applied PCA retaining 95% variance (30 to 17 components)
- Trained and compared Logistic Regression, Random Forest and XGBoost, with and without PCA

## Results (20% held-out test set)

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.803 | 0.572 | 0.607 | 0.836 |
| Random Forest | 0.790 | 0.519 | 0.567 | 0.816 |
| XGBoost | 0.771 | 0.527 | 0.550 | 0.811 |

With PCA applied, accuracy dropped slightly (LR 0.787, RF 0.770, XGB 0.762), which is expected since PCA rotates features in a way tree-based models don't benefit from.

## Note on accuracy
About 73% of customers don't churn, so a model that always predicts "no churn" scores 73%. Recall and ROC-AUC are the more meaningful metrics here, since missing a churner costs more than a false alarm.

## Stack
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn

## Files
- `churn-prediction.ipynb` — full analysis
