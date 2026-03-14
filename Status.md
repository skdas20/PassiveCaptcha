# Status

## Current Work

PassiveCaptcha is running as a demo passive bot-detection system with:

- a React frontend that collects passive behavioral signals
- a FastAPI backend that scores sessions
- imported public training data from the BORDaR web bot dataset
- trained baseline, XGBoost, and LSTM model artifacts

## What Works Now

- live demo page at `http://localhost:5173`
- backend health check at `http://localhost:8000/health`
- browser session scoring through `POST /v1/ingest`
- dataset import into `backend/train/data/sessions.csv`
- saved artifacts in `backend/train/artifacts`

## Run Commands

Frontend:

```bash
cd /workspaces/PassiveCaptcha/apps/web
npm install
npm run dev -- --host 0.0.0.0
```

Backend:

```bash
cd /workspaces/PassiveCaptcha/backend
pip install -r requirements-api.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Training:

```bash
cd /workspaces/PassiveCaptcha/backend
python -m train.import_bordar
pip install -r requirements-ml.txt
python -m train.train_baseline
python -m train.train_xgboost
python -m train.train_lstm
```

## Models Trained

- `RandomForest` baseline
- `XGBoost` calibrated classifier
- `PyTorch` LSTM on mouse sequences

Current best public-data result:

- baseline tabular model is strongest so far

## Work In Progress

- using public datasets to bootstrap the ML layer quickly
- keeping the API ready to serve saved artifacts
- validating which model family is strongest before adding more complexity

## Left Further

- collect app-specific human and bot sessions
- add the rest of the planned passive features beyond mouse-centric public data
- compare baseline vs XGBoost inside the API
- build persistence for labeled sessions
- add dashboard, SHAP, UMAP/t-SNE, Redis, JWT, and fallback challenge
- retrain on richer data and tune thresholds for production use
