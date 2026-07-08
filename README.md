
# Heart Warehouse — Dashboard, ML & Explainability

This package contains a Streamlit app, an ETL script, and a SQLite data warehouse built from `heart.csv`.

## Files
- `warehouse.db` — SQLite data warehouse (staging, dims, fact)
- `etl.py` — ETL script to rebuild warehouse.db from the CSV
- `app.py` — Streamlit app (dashboard, ML models, SHAP explainability)
- `requirements.txt` — Python deps
- `Procfile` — for deployment platforms (Heroku / Streamlit Community Cloud)
- `sample_preview.csv` — sample rows

## Run locally
1. Create venv and install packages:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. (Optional) Rebuild warehouse:
   ```bash
   python etl.py
   ```

3. Run Streamlit:
   ```bash
   streamlit run app.py
   ```

## Notes
- SHAP and XGBoost are optional; the app falls back to Logistic Regression if XGBoost is not available.
- If deploying to Streamlit Community Cloud, include `requirements.txt` and `app.py` at repo root.
