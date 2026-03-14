from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "sessions.csv"
ARTIFACTS_DIR = ROOT / "artifacts"


def find_best_threshold(y_true: pd.Series, positive_scores: list[float]) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, positive_scores, pos_label="human")
    best_index = 0
    best_score = -1.0

    for index, threshold in enumerate(thresholds):
        f1_numerator = 2 * precision[index] * recall[index]
        f1_denominator = precision[index] + recall[index]
        f1_score = 0.0 if f1_denominator == 0 else f1_numerator / f1_denominator
        if f1_score > best_score:
            best_score = f1_score
            best_index = index

    human_threshold = float(thresholds[best_index]) if len(thresholds) else 0.75
    return {
        "human": round(human_threshold, 4),
        "review": round(max(0.25, human_threshold - 0.2), 4),
    }


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset at {DATASET_PATH}. Run `python -m backend.train.import_bordar` first."
        )

    frame = pd.read_csv(DATASET_PATH)
    target = frame["label"]
    features = frame.drop(columns=["label", "sessionId"])

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    calibrated = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
    calibrated.fit(x_train.to_numpy(), (y_train == "human").astype(int))

    positive_scores = calibrated.predict_proba(x_test.to_numpy())[:, 1]
    thresholds = find_best_threshold(y_test, positive_scores.tolist())

    predictions = [
        "human" if score >= thresholds["human"] else "bot"
        for score in positive_scores
    ]
    report = classification_report(y_test, predictions, output_dict=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, ARTIFACTS_DIR / "xgboost_calibrated_model.joblib")
    (ARTIFACTS_DIR / "xgboost_feature_columns.json").write_text(
        json.dumps(list(features.columns), indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "xgboost_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "xgboost_thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )

    print(json.dumps({"thresholds": thresholds, "accuracy": report["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
