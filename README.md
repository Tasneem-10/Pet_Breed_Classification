# 🐶 Pet Breed Classification MLOps Pipeline

https://img.shields.io/badge/Python-3.10-blue]()
https://img.shields.io/badge/PyTorch-Deep%20Learning-red]()
https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue]()
https://img.shields.io/badge/DVC-Data%20Versioning-green]()
https://img.shields.io/badge/Docker-Containerized-2496ED]()
https://img.shields.io/badge/FastAPI-Serving-009688]()
https://img.shields.io/badge/CI-GitHub%20Actions-success]()

End-to-end MLOps project for classifying pet breeds from images using transfer learning and production-grade ML tools.

---

## 🚀 Project Highlights

✅ Transfer Learning with ResNet

✅ Experiment Tracking using MLflow

✅ Model Registry

✅ Dataset & Pipeline Versioning using DVC

✅ CI/CD with GitHub Actions

✅ Model Serving with FastAPI & BentoML

✅ Containerization using Docker

✅ API Load Testing with Locust

✅ Model Optimization (ResNet50 → ResNet18)

✅ Data Drift Monitoring using PSI

---

## 📌 Dataset

**Oxford-IIIT Pet Dataset**

| Metric | Value |
|----------|----------|
| Classes | 37 |
| Total Images (trainval) | 3680 |
| Training Images | 2944 |
| Validation Images | 736 |
| Image Size | 224 × 224 |

Dataset storage and versioning are managed using **DVC**, keeping large files outside Git.

---

## 🏗️ Project Architecture

```text
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

🧠 Model Development
Baseline Model
Architecture: ResNet50
Transfer Learning
Modified Final Layer → 37 Classes
Baseline Performance
Metric	ValueTraining Accuracy	87.98%
Validation Accuracy	70.52%
Average Inference Time	170.12 ms
Locust P95 Latency	2500 ms

The large train-validation gap indicated overfitting.

📊 MLflow Experiment Tracking

Five experiments were conducted using different learning rates:

0.0001
0.0003
0.0005
0.001
0.003

Best Run
Metric	ValueLearning Rate	0.0001
Validation Accuracy	89.81%
Registered Model
Model Name: PetBreedClassifier
Version: 1

🔄 DVC Pipeline
Dataset
   ↓
 Train
   ↓
Evaluate

Pipeline Files
dvc.yaml
dvc.lock
train.py
evaluate.py

Reproduce Pipeline
dvc repro

🌐 FastAPI Serving
Endpoints
GET /health
POST /predict

Example Response
{
  "breed": "Miniature Pinscher",
  "confidence": 0.83
}

🐳 Docker
Build Image
docker build -t pet-breed-api .

Run Container
docker run -p 8001:8000 pet-breed-api

API Docs
http://localhost:8001/docs

✅ Testing

Tests implemented using Pytest:

Health Endpoint
Invalid Request Validation
pytest -v


Result:

2 passed

🔁 Continuous Integration

GitHub Actions automatically performs:

Checkout
   ↓
Setup Python
   ↓
Install Dependencies
   ↓
Run Tests
   ↓
Build Docker Image


Workflow file:

.github/workflows/ci.yml

🚀 Production Serving with BentoML
Endpoint
POST /predict

Local Server
http://localhost:3000

Status
200 OK

⚡ Performance Optimization

The original ResNet50 model was replaced with a lighter ResNet18 architecture.

Model Comparison
Metric	ResNet50	ResNet18Validation Accuracy	70.52%	88.86%
Average Inference Time	170.12 ms	64.18 ms
Improvements

✅ 62% Faster Inference

✅ Higher Validation Accuracy

✅ Smaller Model Size

Saved Model:

resnet18_pet_breed_optimized.pth

📈 Load Testing (Locust)
Baseline - ResNet50
Metric	ValueRequests	5661
Failures	6
Median	1900 ms
P95	2500 ms
P99	3000 ms
RPS	3.7
Optimized - ResNet18
Metric	ValueRequests	238
Failures	0
Median	280 ms
P95	780 ms
P99	2400 ms
Average	384.08 ms
RPS	5.7
📉 Monitoring

Data drift detection is implemented using PSI (Population Stability Index).

Monitored Feature
Image Brightness
Drift Report
Metric	ValueReference Images	500
Current Images	500
PSI	0.1029
Threshold	0.25
Status	✅ No Significant Drift

Artifacts:

monitoring_metrics.json
monitoring_dashboard.json

📂 Project Structure
Pet_Breed_Classification/
│
├── data/
├── service/
├── tests/
├── .github/workflows/
│
├── train.py
├── evaluate.py
├── mlflow_train.py
├── app.py
├── monitoring.py
├── locustfile.py
├── dvc.yaml
├── Dockerfile
└── README.md

🛠️ Tech Stack
Python
PyTorch
Torchvision
FastAPI
BentoML
MLflow
DVC
Docker
GitHub Actions
Locust
Pytest
NumPy
Pillow
⚠️ Limitations
Experiments were performed on CPU.
Results may vary across hardware configurations.
Monitoring is implemented as a local PSI demonstration.
Performance benchmarks were conducted in separate runs.
👩‍💻 Author

Tasneem Hany Mohamed

MLOps Final Project

