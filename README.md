# PassiveCaptcha

PassiveCaptcha is a passive bot-detection demo built in the correct order:
collect session data first, engineer features second, train models third, then serve inference.

## Project structure

- `apps/web`: React + TypeScript demo page and client-side signal collector.
- `backend/app`: FastAPI ingestion and inference service.
- `backend/train`: training entry point, schema, artifacts, and dataset staging area.
- `docker-compose.yml`: local demo stack.

## Development flow

1. Run the web demo and collect behavioral sessions.
2. Send raw session payloads to the FastAPI ingest endpoint.
3. Export engineered feature rows with labels into `backend/train/data/sessions.csv`.
4. Train the baseline model and save artifacts in `backend/train/artifacts`.
5. The API automatically switches to artifact-backed model inference when saved artifacts exist.

## Commands

### Web

```bash
npm install
npm run dev:web
```

### API

```bash
cd backend
pip install -r requirements-api.txt
uvicorn app.main:app --reload
```

### Training

```bash
cd backend
python -m train.import_bordar
pip install -r requirements-ml.txt
python -m train.train_baseline
python -m train.train_xgboost
python -m train.train_lstm
```
