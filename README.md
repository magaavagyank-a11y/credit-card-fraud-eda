# Credit Card Fraud Detection — EDA

Exploratory Data Analysis of the Credit Card Fraud Detection dataset.

## Project Structure
- `notebooks/CreditCardEDA.ipynb` — main EDA notebook
- `data/` — dataset location (not included in repo)

## Main Findings
- Strongly imbalanced dataset (~0.17% fraud)
- Fraud transactions tend to have lower amounts
- Key correlated features: V14, V17, V12, V10

## How to run
```bash
pip install -r requirements.txt
jupyter notebook