# Project 130: E-Commerce CLV and Retention Engine

This project uses past e-commerce transactions to estimate customer lifetime value (CLV) and identify customers who may need retention attention.

## MVP goals

- Inspect and clean transaction data.
- Create customer-level behavior features.
- Train and evaluate a baseline CLV model.
- Estimate retention risk.
- Show results in a dashboard.

## Project folders

- `data/raw` — original dataset files.
- `data/processed` — cleaned data.
- `notebooks` — data exploration and experiments.
- `src` — reusable data and machine learning code.
- `models` — saved trained models.
- `backend` — API for predictions.
- `dashboard` — dashboard app.
- `tests` — checks for the project code.

## Install dependencies

```text
pip install -r requirements.txt