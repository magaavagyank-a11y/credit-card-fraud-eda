# Credit Card Fraud Detection — EDA & Modeling

Exploratory data analysis and baseline machine-learning models for the
[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 492 of them fraud).

## Project Structure

```
├── data/                       # dataset location (not included in the repo)
├── notebooks/
│   └── CreditCardEDA.ipynb     # main notebook: EDA, feature engineering, models
├── requirements.txt
└── README.md
```

## What's Inside

- **EDA**: class imbalance, transaction amount and time distributions, correlation analysis
- **Feature engineering**: hour of day, log amount, time since previous transaction, rolling amount statistics
- **Imbalance handling**: random undersampling vs. class weighting
- **Models**: Decision Tree, Random Forest, XGBoost, Logistic Regression
- **Evaluation**: precision / recall for the fraud class, ROC-AUC vs. PR-AUC

## Main Findings

- The dataset is extremely imbalanced (~0.17% fraud), so accuracy and ROC-AUC are misleading — PR-AUC is the key metric.
- Fraud transactions have a lower median amount (≈9 vs. ≈22) but a long right tail.
- Fraud is relatively more frequent during night (low-traffic) hours.
- The most informative features are **V17, V14, V12 and V10**.

## Results (test set, 20% stratified split)

| Model                          | PR-AUC | ROC-AUC | Precision (fraud) | Recall (fraud) |
|--------------------------------|-------:|--------:|------------------:|---------------:|
| Random Forest                  | 0.864  | 0.974   | 0.94              | 0.82           |
| XGBoost (`scale_pos_weight`)   | 0.838  | 0.973   | 0.50              | 0.85           |
| Logistic Regression (balanced) | 0.716  | 0.972   | 0.06              | 0.92           |
| Decision Tree                  | 0.561  | 0.872   | 0.75              | 0.74           |
| Decision Tree (undersampled)   | 0.015  | 0.906   | 0.02              | 0.91           |

## How to Run

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and put it in `data/`.
2. Install dependencies and start Jupyter:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/CreditCardEDA.ipynb
```

> **macOS note:** XGBoost requires the OpenMP runtime. If you get
> `XGBoostError: libxgboost.dylib could not be loaded`, run `brew install libomp`.
