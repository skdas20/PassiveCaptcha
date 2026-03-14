from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "raw" / "web_bot_detection_dataset"
OUTPUT_PATH = ROOT / "data" / "sessions.csv"

POINT_PATTERN = re.compile(r"\[(?:m)?\((\d+),(\d+)\)\]")
CLICK_PATTERN = re.compile(r"\[c\([^)]+\)\]")


@dataclass
class SessionFeatures:
    session_id: str
    sample_count: int
    mouse_speed_mean: float
    mouse_linearity: float
    click_count: int
    scroll_events: int
    key_events: int
    average_key_interval: float
    visibility_changes: int
    time_to_first_interaction_ms: int
    label: str

    def as_row(self) -> dict[str, str | int | float]:
        return {
            "sessionId": self.session_id,
            "sampleCount": self.sample_count,
            "mouseSpeedMean": round(self.mouse_speed_mean, 6),
            "mouseLinearity": round(self.mouse_linearity, 6),
            "clickCount": self.click_count,
            "scrollEvents": self.scroll_events,
            "keyEvents": self.key_events,
            "averageKeyInterval": round(self.average_key_interval, 6),
            "visibilityChanges": self.visibility_changes,
            "timeToFirstInteractionMs": self.time_to_first_interaction_ms,
            "label": self.label,
        }


def distance(x1: int, y1: int, x2: int, y2: int) -> float:
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def parse_points(behaviour: str) -> list[tuple[int, int]]:
    return [(int(x), int(y)) for x, y in POINT_PATTERN.findall(behaviour)]


def parse_times(times_blob: str | None, expected_points: int) -> list[int]:
    if not times_blob:
        return list(range(expected_points))

    values = [int(value) for value in times_blob.split(",") if value.strip()]
    if len(values) >= expected_points:
        return values[:expected_points]

    if not values:
        return list(range(expected_points))

    last = values[-1]
    while len(values) < expected_points:
        last += 1
        values.append(last)
    return values


def compute_features(
    session_id: str,
    behaviour: str,
    times_blob: str | None,
    label: str,
) -> SessionFeatures:
    points = parse_points(behaviour)
    times = parse_times(times_blob, len(points))
    click_count = len(CLICK_PATTERN.findall(behaviour))

    total_speed = 0.0
    path_distance = 0.0
    direct_distance = 0.0

    for index in range(1, len(points)):
        prev_x, prev_y = points[index - 1]
        curr_x, curr_y = points[index]
        delta_t = max(times[index] - times[index - 1], 1)
        delta_distance = distance(prev_x, prev_y, curr_x, curr_y)
        total_speed += delta_distance / delta_t
        path_distance += delta_distance

    if len(points) > 1:
        direct_distance = distance(points[0][0], points[0][1], points[-1][0], points[-1][1])

    return SessionFeatures(
        session_id=session_id,
        sample_count=len(points),
        mouse_speed_mean=0.0 if len(points) < 2 else total_speed / (len(points) - 1),
        mouse_linearity=0.0 if path_distance == 0 else direct_distance / path_distance,
        click_count=click_count,
        scroll_events=0,
        key_events=0,
        average_key_interval=0.0,
        visibility_changes=0,
        time_to_first_interaction_ms=0,
        label="human" if label == "human" else "bot",
    )


def load_phase1_annotations(annotation_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for split_name in ("train", "test"):
        split_path = annotation_path / split_name
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            session_id, label = line.split()
            labels[session_id] = label
    return labels


def import_phase1_variant(variant_name: str) -> list[SessionFeatures]:
    labels = load_phase1_annotations(RAW_ROOT / "phase1" / "annotations" / variant_name)
    base_dir = RAW_ROOT / "phase1" / "data" / "mouse_movements" / variant_name
    rows: list[SessionFeatures] = []

    for session_id, label in labels.items():
        payload_path = base_dir / session_id / "mouse_movements.json"
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        behaviour = payload.get("total_behaviour") or payload.get("mousemove_total_behaviour", "")
        rows.append(
            compute_features(
                session_id=session_id,
                behaviour=behaviour,
                times_blob=payload.get("mousemove_times"),
                label=label,
            )
        )

    return rows


def load_phase2_annotations(annotation_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        session_id, label = line.split()
        base_session_id = session_id.rsplit("_", 1)[0]
        labels[base_session_id] = label
    return labels


def import_phase2_file(data_path: Path, labels: dict[str, str]) -> list[SessionFeatures]:
    rows: list[SessionFeatures] = []
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            session_id = payload["session_id"]
            label = labels.get(session_id)
            if not label:
                continue
            rows.append(
                compute_features(
                    session_id=session_id,
                    behaviour=payload.get("mousemove_total_behaviour", ""),
                    times_blob=payload.get("mousemove_times"),
                    label=label,
                )
            )
    return rows


def write_output(rows: list[SessionFeatures]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sessionId",
                "sampleCount",
                "mouseSpeedMean",
                "mouseLinearity",
                "clickCount",
                "scrollEvents",
                "keyEvents",
                "averageKeyInterval",
                "visibilityChanges",
                "timeToFirstInteractionMs",
                "label",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())


def main() -> None:
    rows: list[SessionFeatures] = []
    rows.extend(import_phase1_variant("humans_and_moderate_bots"))
    rows.extend(import_phase1_variant("humans_and_advanced_bots"))

    phase2_mixed_labels = load_phase2_annotations(
        RAW_ROOT
        / "phase2"
        / "annotations"
        / "humans_and_moderate_and_advanced_bots"
        / "humans_and_moderate_and_advanced_bots"
    )
    rows.extend(
        import_phase2_file(
            RAW_ROOT
            / "phase2"
            / "data"
            / "mouse_movements"
            / "humans"
            / "mouse_movements_humans.json",
            phase2_mixed_labels,
        )
    )
    rows.extend(
        import_phase2_file(
            RAW_ROOT
            / "phase2"
            / "data"
            / "mouse_movements"
            / "bots"
            / "mouse_movements_moderate_bots.json",
            phase2_mixed_labels,
        )
    )
    rows.extend(
        import_phase2_file(
            RAW_ROOT
            / "phase2"
            / "data"
            / "mouse_movements"
            / "bots"
            / "mouse_movements_advanced_bots.json",
            phase2_mixed_labels,
        )
    )

    deduped_rows = list({row.session_id: row for row in rows}.values())
    write_output(deduped_rows)

    human_count = sum(1 for row in deduped_rows if row.label == "human")
    bot_count = sum(1 for row in deduped_rows if row.label == "bot")
    print(
        f"Wrote {len(deduped_rows)} rows to {OUTPUT_PATH} "
        f"({human_count} human / {bot_count} bot)."
    )


if __name__ == "__main__":
    main()
