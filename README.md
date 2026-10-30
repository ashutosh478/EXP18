# Iris ML CI/CD

A small machine-learning CI/CD example using the provided Iris dataset, scikit-learn, pytest, and GitHub Actions.

## What happens automatically

- On pushes to `main`/`master` and pull requests: install dependencies, train the model, check validation accuracy, run tests, and upload the model files as a workflow artifact.
- The model must reach at least **90% validation accuracy**. If training, the accuracy check, or tests fail, no model is published.
- On a pushed version tag such as `v1.0.0`: run the same checks, then publish the validated model and its metrics as downloadable files in a GitHub Release.

This deployment publishes a versioned model artifact. It does not create a continuously running prediction API or deploy to a cloud server.

## Run locally

Requires Python 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
pytest -q
```

Successful training creates these files in `artifacts/`:

- `iris_model.joblib`: trained scikit-learn pipeline
- `metrics.json`: validation accuracy and quality-gate result
- `model_metadata.json`: feature names, classes, and basic training details

## Enable the pipeline on GitHub

1. Push this project to a GitHub repository with the default branch named `main` or `master`.
2. Open the repository's **Actions** tab and confirm Actions are enabled.
3. Push a code change or open a pull request to trigger CI.
4. To publish a versioned model release after the checks pass, run:

   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

The workflow grants `contents: write` only to the release job so it can create the GitHub Release. If repository or organization policy blocks workflow write permissions, allow the workflow to create releases in the repository's Actions settings.

## Project layout

```text
.
├── .github/workflows/ml-cicd.yml
├── data/Iris.csv
├── tests/test_model.py
├── train.py
├── requirements.txt
└── README.md
```
