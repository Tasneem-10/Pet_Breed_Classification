# Pet Breed Classification

An end-to-end MLOps project for classifying pet breed images using transfer learning with ResNet and production-oriented ML tooling.

---

## Project Overview

This project classifies images from the Oxford-IIIT Pet dataset into 37 pet breed classes.

The project covers the complete machine learning lifecycle:

- Data inspection and visualization
- Data preprocessing
- Transfer learning with ResNet
- Model training and evaluation
- Experiment tracking with MLflow
- Model Registry
- Dataset and pipeline versioning with DVC
- Automated testing with pytest
- Continuous Integration with GitHub Actions
- REST API serving with FastAPI
- Containerization with Docker
- Production-oriented serving with BentoML
- Load testing with Locust
- Model optimization using ResNet18
- Data drift monitoring using PSI

---

## Dataset

The project uses the Oxford-IIIT Pet dataset.

The current experiments use the `trainval` split:

- Images: 3,680
- Classes: 37
- Training images: 2,944
- Validation images: 736

Images are resized to `224 × 224` and converted to tensors.

The dataset is tracked using DVC and is not stored directly in Git.

---

## Baseline Model

The baseline model uses a pretrained **ResNet50**.

The final classification layer was replaced with a linear layer containing 37 output classes.

### Baseline Results

- Training Accuracy: **87.98%**
- Validation Accuracy: **70.52%**
- Average inference time: **170.12 ms**
- Locust p95 latency: **2500 ms**

The difference between training and validation accuracy indicated overfitting in the baseline experiment.

---

## MLflow

MLflow was used for experiment tracking and model management.

Five screening experiments were performed using different learning rates:

```text
0.0001
0.0003
0.0005
0.001
0.003

The best screening run achieved:

Learning rate: 0.0001
Validation Accuracy: 89.81%

The selected model was also registered in the MLflow Model Registry as:

PetBreedClassifier

Model version:

Version 1
DVC

DVC was used to track the dataset and machine learning pipeline.

The pipeline consists of:

Dataset
   ↓
Train
   ↓
Evaluate

The pipeline contains:

train.py
evaluate.py
dvc.yaml
dvc.lock

The pipeline was successfully reproduced using:

dvc repro

DVC keeps the large dataset outside Git while Git tracks the corresponding metadata and pipeline definitions.

FastAPI

A FastAPI application was created with:

GET /health
POST /predict

The /predict endpoint accepts an image and returns the predicted breed and confidence.

Example response:

{
  "breed": "Miniature Pinscher",
  "confidence": 0.83
}

The API was tested using FastAPI's Swagger interface and automated tests.

Testing

Pytest was used for API testing.

Current tests cover:

Health endpoint
Validation failure when no image is provided

Run the tests with:

pytest -v

The tests completed successfully:

2 passed
Docker

The FastAPI service was containerized using Docker.

The Docker image can be built with:

docker build -t pet-breed-api .

The container exposes port 8000.

Example:

docker run -p 8001:8000 pet-breed-api

The API can then be accessed through:

http://localhost:8001/docs
GitHub Actions

GitHub Actions was used to automate Continuous Integration.

The CI pipeline performs:

Checkout
   ↓
Set up Python
   ↓
Install dependencies
   ↓
Run pytest
   ↓
Build Docker image

The workflow is defined in:

.github/workflows/ci.yml

The CI workflow successfully passed after resolving dependencies on the local dataset and model weights.

Production Serving with BentoML

BentoML was used to serve the optimized model as a production-oriented inference service.

The service exposes:

POST /predict

The BentoML server runs locally on:

http://localhost:3000

A real prediction request was successfully tested with HTTP status:

200
Load Testing with Locust

Locust was used to measure API performance.

Baseline — ResNet50
Metric	Result
Requests	5661
Failures	6
Median	1900 ms
p95	2500 ms
p99	3000 ms
RPS	3.7
Optimized — ResNet18
Metric	Result
Requests	238
Failures	0
Median	280 ms
p95	780 ms
p99	2400 ms
Average	384.08 ms
RPS	5.7

The optimized service showed lower p95 latency and zero failures during the measured Locust test.

Latency measurements are local measurements and can vary depending on hardware, concurrency, and runtime conditions.

Model Optimization

The baseline ResNet50 model was replaced with a lighter pretrained ResNet18 architecture.

Model Comparison
Metric	ResNet50	ResNet18
Validation Accuracy	70.52%	88.86%
Average Inference Time	170.12 ms	64.18 ms

The average local inference time decreased from 170.12 ms to 64.18 ms.

The optimized model was saved as:

resnet18_pet_breed_optimized.pth
Monitoring

A data-drift monitoring step was implemented using the Population Stability Index (PSI).

The monitored feature is image brightness.

Monitoring Result
Metric	Result
Reference images	500
Current images	500
PSI	0.1029
Threshold	0.25
Status	NO SIGNIFICANT DRIFT

The monitoring result is saved in:

monitoring_metrics.json

The configured drift threshold is:

0.25

The project also contains:

monitoring_dashboard.json

which documents the monitoring dashboard configuration.

The current monitoring implementation demonstrates PSI-based drift detection locally. It is not a continuously running production monitoring system.

Project Structure
Pet_Breed_Classification/
│
├── data/
│   └── raw/
│       └── oxford-iiit-pet/
│
├── service/
│   └── service.py
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app.py
├── class_names.py
├── download_data.py
├── inspect_data.py
├── visualize_data.py
├── prepare_data.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── mlflow_train.py
├── mlflow_experiments.py
├── register_model.py
├── optimize_model.py
├── benchmark.py
├── monitoring.py
├── monitoring_metrics.json
├── monitoring_dashboard.json
├── locustfile.py
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── Dockerfile
└── README.md

Large generated files such as model weights, the virtual environment, MLflow database files, and the raw dataset are excluded from Git.

Main Technologies
Python
PyTorch
Torchvision
FastAPI
BentoML
MLflow
DVC
Docker
GitHub Actions
pytest
Locust
NumPy
Pillow
Reproducibility

The project uses:

A fixed random seed for train/validation splitting
DVC for dataset and pipeline tracking
dvc.lock for pipeline reproducibility
MLflow for experiment tracking
GitHub Actions for automated testing and Docker builds
Project Workflow
Oxford-IIIT Pet Dataset
          ↓
    Data Inspection
          ↓
     Preprocessing
          ↓
    ResNet50 Baseline
          ↓
        MLflow
          ↓
         DVC
          ↓
   FastAPI + Docker
          ↓
      CI Testing
          ↓
       BentoML
          ↓
    Locust Testing
          ↓
  ResNet18 Optimization
          ↓
   Performance Benchmark
          ↓
    PSI Monitoring
Results Summary

The project progressed from a ResNet50 baseline to an optimized ResNet18 production-oriented pipeline.

The optimized model achieved:

88.86% validation accuracy
64.18 ms average local inference time
780 ms p95 latency in the measured Locust test
0 failures in the optimized Locust test

The monitoring experiment reported:

PSI = 0.1029

which was below the configured threshold:

0.25
Limitations
The current experiments were performed on CPU.
The current experiments use the trainval split of the Oxford-IIIT Pet dataset.
Reported latency values are local measurements and can vary depending on hardware and workload.
The current monitoring implementation demonstrates PSI-based drift detection but does not represent a continuously running production monitoring system.
The optimized Locust test and baseline Locust test were performed in separate runs, so their results should be interpreted as measured local benchmarks rather than a controlled scientific performance comparison.
Author

Tasneem Hany Mohamed

MLOps Final Project