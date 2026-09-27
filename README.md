# 🐶 Pet Breed Classification MLOps Pipeline

End-to-end MLOps project for classifying pet breed images using transfer learning with ResNet and production-oriented machine learning tools.

---

## 🚀 Project Highlights

✅ Transfer Learning with ResNet

✅ Experiment Tracking using MLflow

✅ Model Registry

✅ Dataset & Pipeline Versioning using DVC

✅ REST API Development with FastAPI

✅ Production Serving using BentoML

✅ Docker Containerization

✅ Load Testing with Locust

✅ CI/CD using GitHub Actions

✅ Model Optimization (ResNet50 → ResNet18)

✅ Data Drift Monitoring using PSI

---

## 📌 Dataset

This project uses the Oxford-IIIT Pet Dataset.

| Metric | Value |
|----------|----------|
| Classes | 37 |
| Images (trainval split) | 3,680 |
| Training Images | 2,944 |
| Validation Images | 736 |
| Image Size | 224 × 224 |

The dataset is tracked with DVC and stored outside Git.

---

## 🏗️ Project Architecture

```text
              Oxford-IIIT Pet Dataset
                         │
                         ▼
                 Data Inspection
                         │
                         ▼
                  Preprocessing
                         │
                         ▼
                ResNet50 Baseline
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
          MLflow                    DVC
     Experiment Tracking      Data & Pipeline
       + Model Registry          Versioning
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                  FastAPI Service
                         │
                         ▼
                       Docker
                         │
                         ▼
                 GitHub Actions
                         │
                         ▼
                     BentoML
                         │
                         ▼
                  Locust Testing
                         │
                         ▼
               ResNet18 Optimization
                         │
                         ▼
              Performance Benchmark
                         │
                         ▼
                 PSI Monitoring
```

The architecture demonstrates the complete machine learning lifecycle from dataset management and model training to deployment, testing, optimization, and monitoring.

---

## 🧠 Model Development

### Baseline Model

- Pretrained ResNet50
- Transfer Learning
- Modified Final Layer
- 37 Output Classes

### Baseline Results

| Metric | Value |
|----------|----------|
| Training Accuracy | 87.98% |
| Validation Accuracy | 70.52% |
| Average Inference Time | 170.12 ms |
| Locust p95 Latency | 2500 ms |

The gap between training and validation accuracy indicated overfitting.

---

## 📊 MLflow Experiment Tracking

Five screening experiments were conducted using different learning rates:

- 0.0001
- 0.0003
- 0.0005
- 0.001
- 0.003

All experiments used the same dataset and the same 80/20 train-validation split with a fixed random seed.

### Best Run

| Metric | Value |
|----------|----------|
| Learning Rate | 0.0001 |
| Validation Accuracy | 89.81% |

These experiments were used for hyperparameter screening rather than as the final production training run.

### Registered Model

- Model Name: PetBreedClassifier
- Version: 1

---

## 🔄 DVC Pipeline

Pipeline Structure:

```text
Dataset
   │
   ▼
 Train
   │
   ▼
Evaluate
```

Pipeline Files:

- `train.py`
- `evaluate.py`
- `dvc.yaml`
- `dvc.lock`

Reproduce pipeline:

```bash
dvc repro
```

DVC is used to track the dataset and reproduce the training and evaluation pipeline.

---

## 🌐 FastAPI Service

### Endpoints

```http
GET /health
POST /predict
```

### Example Response

```json
{
  "breed": "Miniature Pinscher",
  "confidence": 0.83
}
```

The API was tested locally using FastAPI Swagger documentation.

---

## ✅ Testing

Implemented using **Pytest**.

### Covered Tests

- Health Endpoint
- Invalid Request Validation

### Run Tests

```bash
pytest -v
```

### Result

```text
2 passed
```

---

## 🐳 Docker

### Build Image

```bash
docker build -t pet-breed-api .
```

### Run Container

```bash
docker run -p 8001:8000 pet-breed-api
```

### API Documentation

```text
http://localhost:8001/docs
```

The Docker image was successfully built and the FastAPI service was tested inside the container.

---

## 🔁 Continuous Integration

GitHub Actions Workflow:

```text
Checkout Repository
         │
         ▼
     Setup Python
         │
         ▼
Install Dependencies
         │
         ▼
      Run Tests
         │
         ▼
  Build Docker Image
```

### Workflow File

`.github/workflows/ci.yml`

The CI workflow automatically runs tests and verifies that the Docker image can be built successfully.

---

## 🚀 Production Serving with BentoML

### Endpoint

```http
POST /predict
```

### Local Server

```text
http://localhost:3000
```

### Test Result

```text
Status code: 200
```

BentoML was used to serve the trained model as a production-oriented inference service.

---

## ⚡ Model Optimization

The baseline **ResNet50** model was replaced with a lighter **ResNet18** architecture.

### Model Comparison

| Metric | ResNet50 | ResNet18 |
|----------|----------|----------|
| Validation Accuracy | 70.52% | 88.86% |
| Average Inference Time | 170.12 ms | 64.18 ms |
| Locust p95 Latency | 2500 ms | 780 ms |
| Locust Failures | 6 | 0 |

### Improvements

✅ Higher Validation Accuracy

✅ Faster Inference

✅ Lower Latency in the measured load test

✅ Better Deployment Performance

### Optimized Model

```text
resnet18_pet_breed_optimized.pth
```

The optimization reduced the measured average inference time from **170.12 ms** to **64.18 ms**.

---

## 📈 Load Testing (Locust)

### Baseline - ResNet50

| Metric | Value |
|----------|----------|
| Requests | 5661 |
| Failures | 6 |
| Median Latency | 1900 ms |
| p95 | 2500 ms |
| p99 | 3000 ms |
| RPS | 3.7 |

### Optimized - ResNet18

| Metric | Value |
|----------|----------|
| Requests | 238 |
| Failures | 0 |
| Median Latency | 280 ms |
| p95 | 780 ms |
| p99 | 2400 ms |
| Average | 384.08 ms |
| RPS | 5.7 |

> Note: The baseline and optimized Locust tests were performed in separate runs, so the results are reported as measured rather than as a strictly controlled benchmark.

---

## 📉 Monitoring

Data Drift Detection is implemented using **Population Stability Index (PSI)**.

### Monitored Feature

- Image Brightness

### Drift Report

| Metric | Value |
|----------|----------|
| Reference Images | 500 |
| Current Images | 500 |
| PSI | 0.1029 |
| Threshold | 0.25 |
| Status | ✅ No Significant Drift |

### Artifacts

```text
monitoring_metrics.json
monitoring_dashboard.json
```

The monitoring implementation is a local PSI-based demonstration rather than a continuously running production monitoring system.

---

## 📂 Project Structure

```text
Pet_Breed_Classification/
│
├── data/
├── service/
├── tests/
├── .github/
│   └── workflows/
│
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── mlflow_train.py
├── register_model.py
├── optimize_model.py
├── benchmark.py
├── monitoring.py
├── monitoring_dashboard.json
├── monitoring_metrics.json
├── locustfile.py
├── dvc.yaml
├── dvc.lock
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

- Python
- PyTorch
- Torchvision
- FastAPI
- BentoML
- MLflow
- DVC
- Docker
- GitHub Actions
- Pytest
- Locust
- NumPy
- Pillow

---

## 🔬 Reproducibility

The project uses:

- Fixed random seed
- DVC pipeline tracking
- DVC dataset versioning
- MLflow experiment tracking
- GitHub Actions CI workflow

The DVC pipeline can be reproduced using:

```bash
dvc repro
```

---

## 📋 Results Summary

### Final Optimized Model

| Metric | Result |
|----------|----------|
| Validation Accuracy | 88.86% |
| Average Inference Time | 64.18 ms |
| p95 Latency | 780 ms |
| Failures | 0 |
| PSI Score | 0.1029 |

---

## ⚠️ Limitations

- Experiments were performed on CPU.
- Results may vary depending on hardware.
- The current dataset experiments use the `trainval` split available through the torchvision dataset interface.
- Monitoring is a local PSI-based demonstration rather than a continuously running production monitoring system.
- Baseline and optimized Locust tests were executed separately.
- Latency values are local measurements.
- The DVC remote storage used in the course environment was not available locally, so remote artifact retrieval was not used for the final demonstration.
- The current CI workflow validates tests and Docker image building; deployment is not automated.

---

## 👩‍💻 Author

**Tasneem Hany Mohamed**

*MLOps Final Project*
