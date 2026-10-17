# Pet Breed Classification MLOps Pipeline

An end-to-end MLOps project for classifying pet breeds using transfer learning with ResNet architectures. The project explores data and pipeline versioning, experiment tracking, model serving, containerization, testing, benchmarking, and basic drift monitoring.

## Table of Contents

* [Project Highlights](#project-highlights)
* [Dataset](#dataset)
* [Quickstart](#quickstart)
* [Model Development](#model-development)
* [Project Architecture](#project-architecture)
* [MLflow Experiment Tracking](#mlflow-experiment-tracking)
* [DVC Pipeline](#dvc-pipeline)
* [FastAPI Service](#fastapi-service)
* [BentoML Service](#bentoml-service)
* [Testing](#testing)
* [Docker](#docker)
* [Load Testing](#load-testing)
* [Monitoring](#monitoring)
* [Project Structure](#project-structure)
* [Limitations](#limitations)
* [Technology Stack](#technology-stack)
* [Author](#author)

## Project Highlights

* Transfer learning with ResNet50 and ResNet18
* Experiment tracking with MLflow
* Model registry experiments
* Dataset and pipeline versioning with DVC
* FastAPI inference service
* BentoML inference service
* Docker containerization
* Automated tests with Pytest
* Load testing with Locust
* GitHub Actions CI
* Basic Population Stability Index (PSI) monitoring

## Dataset

This project uses the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which contains images from 37 cat and dog breeds.

The project uses the `trainval` split provided by `torchvision.datasets.OxfordIIITPet`. The training and validation subsets used by the project depend on the preprocessing and split logic implemented in the training scripts.

### Download the dataset

From the repository root, activate your Python environment and install the project dependencies. Then run:

```bash
python download_data.py
```

The script downloads the dataset into `data/raw` using Torchvision. Internet access is required if the dataset is not already available locally.

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Tasneem-10/Pet_Breed_Classification.git
cd Pet_Breed_Classification
```

### 2. Install dependencies

Use Python 3.11 for the Docker image, or a Python version supported by the installed PyTorch and Torchvision packages.

Create and activate a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies needed by the API and model:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Obtain the trained model

The trained `.pth` files are not included in Git. The optimized API model is expected at:

```text
resnet18_pet_breed_optimized.pth
```

Place the model file in the repository root before starting the API. Obtain it from the project's model artifacts or train/export a compatible checkpoint using the project scripts.

The model must be a compatible ResNet18 state dictionary with a 37-class output layer.

> **Important:** A fresh clone cannot make predictions until a compatible model checkpoint is available. The repository currently does not document a configured model-artifact download URL.

### 4. Download data if needed

```bash
python download_data.py
```

### 5. Start the FastAPI service

```bash
uvicorn app:app --host 127.0.0.1 --port 8001
```

Open the interactive API documentation:

```text
http://127.0.0.1:8001/docs
```

Check the health endpoint:

```text
http://127.0.0.1:8001/health
```

To make a prediction, send a `POST` request to `/predict` using multipart form data with the field name `file`.

Example with curl:

```bash
curl -X POST "http://127.0.0.1:8001/predict" -F "file=@path/to/pet_image.jpg"
```

Example response:

```json
{
  "breed": "Abyssinian",
  "confidence": 0.97
}
```

The breed and confidence depend on the uploaded image and the selected model checkpoint.

## Project Architecture

```text
Oxford-IIIT Pet Dataset
          |
          v
Data Inspection and Preprocessing
          |
          v
Model Training and Evaluation
          |
          +----------> DVC Pipeline
          |
          +----------> MLflow Experiments
          |
          v
Model Checkpoint
          |
          +----------> FastAPI
          |
          +----------> BentoML
          |
          v
Docker / Local Serving
          |
          v
Testing and Benchmarking
          |
          v
PSI Monitoring Demonstration
```

## Model Development

The project includes ResNet50 and ResNet18 model implementations with a final classification layer for 37 pet breeds.

The repository contains training, evaluation, prediction, and optimization scripts, including:

* `train.py`
* `evaluate.py`
* `predict.py`
* `optimize_model.py`
* `benchmark.py`

### Model comparison

The README previously reported the following results:

| Metric                      |  ResNet50 | ResNet18 |
| --------------------------- | --------: | -------: |
| Validation accuracy         |    70.52% |   88.86% |
| Average inference time      | 170.12 ms | 64.18 ms |
| Reported Locust p95 latency |   2500 ms |   780 ms |

These values are historical reported measurements and have not yet been revalidated as a controlled comparison. The models were trained with different training configurations, and the benchmark/load-test conditions may differ. Therefore, the accuracy difference should not be attributed solely to the architecture.

See the Limitations section before interpreting these results.

## MLflow Experiment Tracking

The project includes MLflow experiment tracking scripts for recording model-training experiments and comparing learning rates.

Relevant files:

* `mlflow_train.py`
* `mlflow_experiments.py`
* `register_model.py`

Run an experiment only after installing the relevant dependencies and verifying the script's configuration:

```bash
python mlflow_train.py
```

The exact experiment configuration and output depend on the script and the local MLflow tracking setup. Model registration requires a reachable tracking store and the appropriate artifacts.

## DVC Pipeline

The repository includes:

* `dvc.yaml`
* `dvc.lock`
* `.dvc/`

To inspect the pipeline and attempt reproduction:

```bash
dvc status
dvc repro
```

**Remote-storage limitation:** The repository currently has no configured DVC remote. Consequently, a fresh clone cannot retrieve remote-tracked data or model artifacts using `dvc pull` unless a remote is configured and access is available.

Download the dataset separately with:

```bash
python download_data.py
```

Pipeline reproduction also requires the dependencies, inputs, and model/data artifacts expected by the pipeline stages.

## FastAPI Service

The API is implemented in `app.py`.

### Endpoints

| Method | Endpoint   | Purpose                                 |
| ------ | ---------- | --------------------------------------- |
| GET    | `/health`  | Reports service/model health            |
| POST   | `/predict` | Predicts a breed from an uploaded image |

The prediction endpoint expects an image upload in the multipart form field named `file`.

Run locally:

```bash
uvicorn app:app --host 127.0.0.1 --port 8001
```

Interactive documentation:

```text
http://127.0.0.1:8001/docs
```

The model checkpoint must be present at the path configured in `app.py`.

## BentoML Service

The BentoML service is defined in `service/service.py` and uses the optimized ResNet18 checkpoint.

The service currently expects the model file:

```text
resnet18_pet_breed_optimized.pth
```

The checkpoint must be accessible at the path used by `service/service.py`. The service also requires BentoML, PyTorch, Torchvision, Pillow, and the shared class-name module.

Because BentoML CLI options can vary by version, first install the dependencies and check the installed CLI:

```bash
bentoml --help
```

Then use the serving command supported by the installed BentoML version and the service definition. Verify the exact command against the project's installed BentoML version before relying on it in a fresh environment.

## Testing

Tests are implemented using Pytest.

Install the dependencies, ensure the model checkpoint required by the application is available, and run:

```bash
pytest -v
```

The current tests cover the health endpoint, request validation when the prediction request omits its image file, and successful prediction using a valid local image. The prediction test verifies the response status, returned breed name, and confidence range; it does not measure classification accuracy.

A successful test run does not by itself validate model accuracy or production readiness.

## Docker

The repository contains a `Dockerfile` for the FastAPI application.

Build the image:

```bash
docker build -t pet-breed-api .
```

Run the container:

```bash
docker run --rm -p 8001:8000 pet-breed-api
```

Open the API documentation:

```text
http://localhost:8001/docs
```

**Model checkpoint:** The Docker image copies `resnet18_pet_breed_optimized.pth` into the image and loads it during application startup. The container was tested successfully with the `/health` and `/predict` endpoints. However, the checkpoint is not tracked by Git, and a fresh clone must obtain the checkpoint separately before building the image. No public model-download URL is currently configured.


## Load Testing

The repository includes `locustfile.py` for load testing.

Before running a load test, verify which service the Locust script targets and which multipart field it sends. The FastAPI endpoint expects the field `file`, while the BentoML endpoint may use a different request format.

Do not compare load-test measurements unless the target service, model, input images, concurrency, duration, warm-up, and machine conditions are documented consistently.

Previously reported results were collected in separate runs and should be treated as observations rather than a controlled benchmark.

## Monitoring

The project includes a local PSI-based drift demonstration:

* `monitoring.py`
* `monitoring_metrics.json`
* `monitoring_dashboard.json`

The README previously reported a PSI value of `0.1029` and a threshold of `0.25`. These values should be interpreted as results from the current local demonstration, not evidence of a live production monitoring system.

A production implementation would require a real metrics source, a clearly defined reference/current data window, and integration between the drift calculation and the dashboard.

## Project Structure

```text
Pet_Breed_Classification/
├── .github/workflows/
├── data/
├── service/
│   └── service.py
├── tests/
├── app.py
├── benchmark.py
├── class_names.py
├── download_data.py
├── dvc.yaml
├── dvc.lock
├── evaluate.py
├── locustfile.py
├── mlflow_experiments.py
├── mlflow_train.py
├── model.py
├── monitoring.py
├── optimize_model.py
├── predict.py
├── prepare_data.py
├── register_model.py
├── requirements.txt
├── train.py
└── README.md
```

## Technology Stack

* Python
* PyTorch and Torchvision
* FastAPI
* BentoML
* MLflow
* DVC
* Docker
* GitHub Actions
* Pytest
* Locust
* NumPy
* Pillow

## Limitations

* Experiments were run on CPU, so performance may vary on other hardware.
* The model checkpoint files are not currently distributed through a documented artifact URL.
* The DVC remote is not configured in the repository, preventing remote artifact retrieval from a fresh clone.
* Reported ResNet50 and ResNet18 results were obtained using different training configurations and have not been verified as a controlled architecture comparison.
* The reported metrics should be reconciled with the committed `metrics.json` and the actual evaluation split before being presented as final results.
* The monitoring implementation is a local PSI demonstration, not a continuous production monitoring system.
* The dashboard currently needs a real metrics source to provide live monitoring.
* The CI workflow validates the configured checks and image build; it does not perform automated deployment.

## Author

**Tasneem Hany Mohamed**

*MLOps Final Project*
