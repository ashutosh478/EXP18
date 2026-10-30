import app as prediction_app

from train import MODEL_FILE, train_model


def test_home_route():
    client = prediction_app.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert response.json["endpoint"] == "POST /predict"


def test_valid_prediction(tmp_path, monkeypatch):
    train_model(output_dir=tmp_path)

    monkeypatch.setattr(
        prediction_app,
        "MODEL_PATH",
        tmp_path / MODEL_FILE,
    )

    client = prediction_app.app.test_client()

    response = client.post(
        "/predict",
        json={
            "features": {
                "SepalLengthCm": 5.1,
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2,
            }
        },
    )

    assert response.status_code == 200
    assert response.json["prediction"] == "Iris-setosa"


def test_missing_measurement_is_rejected():
    client = prediction_app.app.test_client()

    response = client.post(
        "/predict",
        json={
            "features": {
                "SepalLengthCm": 5.1,
            }
        },
    )

    assert response.status_code == 400


def test_non_numeric_measurement_is_rejected():
    client = prediction_app.app.test_client()

    response = client.post(
        "/predict",
        json={
            "features": {
                "SepalLengthCm": "invalid",
                "SepalWidthCm": 3.5,
                "PetalLengthCm": 1.4,
                "PetalWidthCm": 0.2,
            }
        },
    )

    assert response.status_code == 400