from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "sessions.csv"
ARTIFACTS_DIR = ROOT / "artifacts"


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset at {DATASET_PATH}. Export feature rows there before training."
        )

    frame = pd.read_csv(DATASET_PATH)
    target = frame["label"]
    features = frame.drop(columns=["label", "sessionId"])

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42
    )
    model.fit(x_train.to_numpy(), y_train)

    predictions = model.predict(x_test.to_numpy())
    report = classification_report(y_test, predictions, output_dict=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "baseline_model.joblib")
    (ARTIFACTS_DIR / "feature_columns.json").write_text(
        json.dumps(list(features.columns), indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "thresholds.json").write_text(
        json.dumps({"human": 0.75, "review": 0.45}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
