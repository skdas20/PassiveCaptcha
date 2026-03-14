from ..schemas import FeatureResponse, InferenceResponse
from .model import feature_row, load_artifacts


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


def score_features(features: FeatureResponse) -> InferenceResponse:
    artifacts = load_artifacts()
    if artifacts is not None:
        model, columns, thresholds = artifacts
        probability = float(model.predict_proba([feature_row(features, columns)])[0][1])
        flagged_signals: list[str] = []

        if features.mouseLinearity > 0.98:
            flagged_signals.append("mouse_linearity")
        if 0 < features.averageKeyInterval < 20:
            flagged_signals.append("typing_rhythm")
        if (
            features.timeToFirstInteractionMs is not None
            and features.timeToFirstInteractionMs < 300
        ):
            flagged_signals.append("time_to_first_interaction")

        human_threshold = float(thresholds.get("human", 0.75))
        review_threshold = float(thresholds.get("review", 0.45))
        verdict = "review"
        if probability >= human_threshold:
            verdict = "human"
        elif probability <= review_threshold:
            verdict = "bot"

        return InferenceResponse(
            score=probability,
            verdict=verdict,
            flaggedSignals=flagged_signals,
            features=features,
        )

    score = 0.5
    flagged_signals: list[str] = []

    if features.sampleCount > 20:
        score += 0.1

    if features.mouseLinearity < 0.92:
        score += 0.12
    elif features.mouseLinearity > 0.98:
        score -= 0.18
        flagged_signals.append("mouse_linearity")

    if features.averageKeyInterval > 40:
        score += 0.08
    elif 0 < features.averageKeyInterval < 20:
        score -= 0.1
        flagged_signals.append("typing_rhythm")

    if (
        features.timeToFirstInteractionMs is not None
        and features.timeToFirstInteractionMs < 300
    ):
        score -= 0.08
        flagged_signals.append("time_to_first_interaction")

    if features.visibilityChanges > 3:
        score -= 0.04
        flagged_signals.append("tab_visibility")

    bounded_score = _clamp(score, 0.01, 0.99)
    verdict = "review"
    if bounded_score >= 0.75:
        verdict = "human"
    elif bounded_score <= 0.45:
        verdict = "bot"

    return InferenceResponse(
        score=bounded_score,
        verdict=verdict,
        flaggedSignals=flagged_signals,
        features=features,
    )
