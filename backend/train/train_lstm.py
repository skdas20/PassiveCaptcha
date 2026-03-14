from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "raw" / "web_bot_detection_dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
POINT_PATTERN = re.compile(r"\[m\((\d+),(\d+)\)\]")


def parse_points(behaviour: str, limit: int = 128) -> list[tuple[float, float]]:
    points = [(float(x), float(y)) for x, y in POINT_PATTERN.findall(behaviour)]
    if not points:
        return [(0.0, 0.0)]
    if len(points) >= limit:
        step = len(points) / limit
        return [points[int(index * step)] for index in range(limit)]
    return points


def normalize_points(points: list[tuple[float, float]], limit: int = 128) -> list[list[float]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    normalized = [[(x - min_x) / range_x, (y - min_y) / range_y] for x, y in points]
    while len(normalized) < limit:
        normalized.append(normalized[-1])
    return normalized[:limit]


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[list[list[float]]], labels: list[int]) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 48) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        logits = self.head(hidden[-1])
        return logits.squeeze(1)


def load_phase2_annotations(annotation_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        session_id, label = line.split()
        labels[session_id.rsplit("_", 1)[0]] = label
    return labels


def load_phase1_annotations(annotation_dir: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for split_name in ("train", "test"):
        for line in (annotation_dir / split_name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            session_id, label = line.split()
            labels[session_id] = label
    return labels


def load_phase1_variant(variant_name: str) -> list[tuple[str, list[list[float]], int]]:
    labels = load_phase1_annotations(RAW_ROOT / "phase1" / "annotations" / variant_name)
    base_dir = RAW_ROOT / "phase1" / "data" / "mouse_movements" / variant_name
    rows: list[tuple[str, list[list[float]], int]] = []

    for session_id, label in labels.items():
        payload_path = base_dir / session_id / "mouse_movements.json"
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        behaviour = payload.get("total_behaviour") or payload.get("mousemove_total_behaviour", "")
        points = parse_points(behaviour)
        rows.append((session_id, normalize_points(points), 1 if label == "human" else 0))

    return rows


def load_sequences(data_path: Path, labels: dict[str, str]) -> list[tuple[str, list[list[float]], int]]:
    rows: list[tuple[str, list[list[float]], int]] = []
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            session_id = payload["session_id"]
            label = labels.get(session_id)
            if not label:
                continue
            points = parse_points(payload.get("mousemove_total_behaviour", ""))
            rows.append(
                (
                    session_id,
                    normalize_points(points),
                    1 if label == "human" else 0,
                )
            )
    return rows


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)

    labels = load_phase2_annotations(
        RAW_ROOT
        / "phase2"
        / "annotations"
        / "humans_and_moderate_and_advanced_bots"
        / "humans_and_moderate_and_advanced_bots"
    )

    rows: list[tuple[str, list[list[float]], int]] = []
    rows.extend(load_phase1_variant("humans_and_moderate_bots"))
    rows.extend(load_phase1_variant("humans_and_advanced_bots"))
    rows.extend(
        load_sequences(
            RAW_ROOT
            / "phase2"
            / "data"
            / "mouse_movements"
            / "humans"
            / "mouse_movements_humans.json",
            labels,
        )
    )
    rows.extend(
        load_sequences(
            RAW_ROOT
            / "phase2"
            / "data"
            / "mouse_movements"
            / "bots"
            / "mouse_movements_moderate_bots.json",
            labels,
        )
    )
    rows.extend(
        load_sequences(
            RAW_ROOT
            / "phase2"
            / "data"
            / "mouse_movements"
            / "bots"
            / "mouse_movements_advanced_bots.json",
            labels,
        )
    )

    deduped = list({session_id: (sequence, label) for session_id, sequence, label in rows}.items())
    session_ids = [session_id for session_id, _ in deduped]
    sequences = [payload[0] for _, payload in deduped]
    y_values = [payload[1] for _, payload in deduped]

    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        sequences,
        y_values,
        session_ids,
        test_size=0.25,
        random_state=42,
        stratify=y_values,
    )

    train_loader = DataLoader(SequenceDataset(x_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(SequenceDataset(x_test, y_test), batch_size=32)

    model = LSTMClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    positive_count = sum(y_train)
    negative_count = len(y_train) - positive_count
    pos_weight = torch.tensor(
        [negative_count / max(positive_count, 1)],
        dtype=torch.float32,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _epoch in range(16):
        model.train()
        for batch_sequences, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_sequences)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions: list[str] = []
    truth: list[str] = ["human" if value == 1 else "bot" for value in y_test]

    with torch.no_grad():
        for batch_sequences, _batch_labels in test_loader:
            probs = torch.sigmoid(model(batch_sequences))
            predictions.extend(["human" if value >= 0.5 else "bot" for value in probs.tolist()])

    report = classification_report(truth, predictions, output_dict=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS_DIR / "lstm_mouse_model.pt")
    (ARTIFACTS_DIR / "lstm_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "lstm_sequence_config.json").write_text(
        json.dumps({"sequence_length": 128, "input_size": 2}, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "accuracy": report["accuracy"],
                "train_sessions": len(id_train),
                "test_sessions": len(id_test),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
