# Premier League Audience Predictor

A SPORTFIVE internal tool for estimating Premier League TV audiences globally and by region.

## What it does

Predicts global and regional TV audiences for any Premier League fixture, with P10/P50/P90 confidence bands. Supports:

- Single-fixture predictions
- Manual basket builder
- Bulk CSV/Excel upload for fixture lists

## How it works

The model is a gradient boosting regressor trained on five seasons of Premier League viewership data (2021/22 - 2025/26). Predictions account for team identity, kick-off slot, concurrent fixtures, prior-season strength, regional time-of-day, and more.

## Live deployment

Hosted via Streamlit Community Cloud.

## Files

- `app_v2.py` — the web app
- `predict.py` — inference module
- `model_v3_clean.pkl` — trained model
- `league_tables.xlsx` — historic league tables for prior-season lookups
- `requirements.txt` — Python dependencies

## Local development

```bash
pip install -r requirements.txt
streamlit run app_v2.py
```
