"""Basic checks for the training and model artifact pipeline."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from train import DATA_PATH, FEATURES, METADATA_FILE, METRICS_FILE, MODEL_FILE, load_dataset, train_model


def test_training_passes_quality_gate_and_saves_model(tmp_path: Path) -> None:
    metrics = train_model(output_dir=tmp_path)

    assert metrics["quality_gate_passed"] is True
    assert metrics["validation_accuracy"] >= metrics["minimum_required_accuracy"]

    model_path = tmp_path / MODEL_FILE
    assert model_path.is_file()
    model = joblib.load(model_path)

    X, _ = load_dataset()
    predictions = model.predict(X.head(3))
    assert len(predictions) == 3

    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["features"] == FEATURES
    saved_metrics = json.loads((tmp_path / METRICS_FILE).read_text(encoding="utf-8"))
    assert saved_metrics["quality_gate_passed"] is True


def test_dataset_with_missing_feature_is_rejected(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "SepalLengthCm": [5.1],
            "SepalWidthCm": [3.5],
            "PetalLengthCm": [1.4],
            "Species": ["Iris-setosa"],
        }
    ).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset(bad_csv)
