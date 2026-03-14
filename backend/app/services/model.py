from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from ..schemas import FeatureResponse


ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "train" / "artifacts"


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Any, list[str], dict[str, float]] | None:
    model_path = ARTIFACTS_DIR / "baseline_model.joblib"
    columns_path = ARTIFACTS_DIR / "feature_columns.json"
    thresholds_path = ARTIFACTS_DIR / "thresholds.json"

    if not (model_path.exists() and columns_path.exists() and thresholds_path.exists()):
        return None

    model = joblib.load(model_path)
    columns = json.loads(columns_path.read_text(encoding="utf-8"))
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    return model, columns, thresholds


def feature_row(features: FeatureResponse, columns: list[str]) -> list[float]:
    values = features.model_dump()
    return [float(values[column]) for column in columns]
