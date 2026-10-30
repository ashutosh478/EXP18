"""Train and validate a small Iris classification model."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Iris.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_FILE = "iris_model.joblib"
METRICS_FILE = "metrics.json"
METADATA_FILE = "model_metadata.json"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"
MIN_ACCURACY = 0.90
RANDOM_STATE = 42


def load_dataset(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and check its required columns."""
    data = pd.read_csv(data_path)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[required].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns")
    return data[FEATURES], data[TARGET]


def train_model(
    data_path: Path = DATA_PATH,
    output_dir: Path = ARTIFACT_DIR,
) -> dict:
    """Train, validate, enforce the accuracy gate, and save release files."""
    X, y = load_dataset(data_path)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    accuracy = float(accuracy_score(y_valid, predictions))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILE
    metadata_path = output_dir / METADATA_FILE
    metrics_path = output_dir / METRICS_FILE

    # Remove any old model first so a failed quality gate cannot publish a stale model.
    for path in (model_path, metadata_path):
        if path.exists():
            path.unlink()

    metrics = {
        "validation_accuracy": round(accuracy, 4),
        "minimum_required_accuracy": MIN_ACCURACY,
        "validation_samples": int(len(X_valid)),
        "quality_gate_passed": accuracy >= MIN_ACCURACY,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if accuracy < MIN_ACCURACY:
        raise RuntimeError(
            f"Model quality gate failed: accuracy {accuracy:.3f} is below {MIN_ACCURACY:.2f}"
        )

    joblib.dump(model, model_path)
    metadata = {
        "model_name": "iris_classifier",
        "algorithm": "StandardScaler + LogisticRegression",
        "features": FEATURES,
        "target": TARGET,
        "classes": sorted(str(label) for label in y.unique()),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_valid)),
        "random_state": RANDOM_STATE,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Quality gate passed (minimum {MIN_ACCURACY:.0%}).")
    print(f"Model saved to: {model_path}")
    return metrics


if __name__ == "__main__":
    train_model()
