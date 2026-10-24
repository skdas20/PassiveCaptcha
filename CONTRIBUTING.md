# Contributing to PassiveCaptcha

Thanks for your interest! PassiveCaptcha sits where machine learning, web security and frontend engineering meet, so there's something for almost every skill set.

## Good places to start

- Issues labelled [`good first issue`](https://github.com/skdas20/PassiveCaptcha/labels/good%20first%20issue) are small and well scoped.
- [`help wanted`](https://github.com/skdas20/PassiveCaptcha/labels/help%20wanted) issues are larger: the SDK, Redis, the dataset and model work.
- Comment on an issue before starting, so work isn't duplicated.

## Local setup

```bash
git clone https://github.com/<your-username>/PassiveCaptcha.git
cd PassiveCaptcha

# Backend (Python 3.12+)
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-api.txt
python -m uvicorn app.main:app --reload --port 8000

# Frontend (Node 20+), in another terminal
cd apps/web
npm install
npm run dev
```

For model training, also install `requirements-ml.txt`; see the README's **Models** section.

## Where things live

| Area | Files |
| --- | --- |
| Signal collection (browser) | `apps/web/src/lib/signalCollector.ts`, `apps/web/src/lib/features.ts` |
| Dashboard and challenge UI | `apps/web/src/Dashboard.tsx`, `apps/web/src/Challenge.tsx` |
| API, auth and rate limiting | `backend/app/main.py`, `backend/app/services/auth.py` |
| Feature extraction | `backend/app/services/features.py` |
| Scoring, ensemble and SHAP | `backend/app/services/scoring.py`, `backend/app/services/model.py` |
| Training | `backend/train/` |

## Pull requests

- Keep each PR focused on one issue and link it (`Fixes #12`).
- Run `npm run build:web` from the repo root before pushing frontend changes.
- For backend changes, include a test or explain how you verified the behaviour.
- **Feature or model changes:** keep `signal_schema.json`, `features.py` and the training scripts in sync, and report before/after metrics in the PR.
- **Privacy rule:** never collect key values, form contents, or anything that identifies a person.

## Code of conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).
