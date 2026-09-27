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

### Best Run

| Metric | Value |
|----------|----------|
| Learning Rate | 0.0001 |
| Validation Accuracy | 89.81% |

### Registered Model

- Model Name: PetBreedClassifier
- Version: 1

---

## 🔄 DVC Pipeline

Pipeline Structure:

    Dataset
       ↓
      Train
       ↓
    Evaluate

Pipeline Files:

- train.py
- evaluate.py
- dvc.yaml
- dvc.lock

Reproduce pipeline:

```bash
dvc repro

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

---

## 🔁 Continuous Integration

GitHub Actions Workflow:

```text
Checkout Repository
         ↓
     Setup Python
         ↓
Install Dependencies
         ↓
      Run Tests
         ↓
  Build Docker Image
```

### Workflow File

```text
.github/workflows/ci.yml
```

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

### Status

```text
200 OK
```

---

## ⚡ Model Optimization

The baseline **ResNet50** model was replaced with a lighter **ResNet18** architecture.

### Model Comparison

| Metric | ResNet50 | ResNet18 |
|----------|----------|----------|
| Validation Accuracy | 70.52% | 88.86% |
| Average Inference Time | 170.12 ms | 64.18 ms |

### Improvements

✅ Higher Validation Accuracy

✅ Faster Inference

✅ Lower Latency

✅ Better Deployment Performance

### Optimized Model

```text
resnet18_pet_breed_optimized.pth
```

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
- Monitoring is a local PSI-based demonstration.
- Benchmark runs were executed separately.
- Latency values are local measurements.

---

## 👩‍💻 Author

**Tasneem Hany Mohamed**

*MLOps Final Project*
