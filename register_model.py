import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
from torch import nn
from torchvision import models


# MLflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment = mlflow.get_experiment_by_name(
    "Pet_Breed_Classification"
)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_accuracy DESC"]
)

best_run = runs.iloc[0]

best_run_id = best_run["run_id"]

print("Best Run ID:", best_run_id)
print(
    "Best Validation Accuracy:",
    best_run["metrics.val_accuracy"]
)


# Rebuild model architecture

NUM_CLASSES = 37

model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


# Load best weights

weights_path = "resnet50_lr_0.0001.pth"

model.load_state_dict(
    torch.load(
        weights_path,
        map_location="cpu"
    )
)

model.eval()


# Log as MLflow Model

with mlflow.start_run(run_id=best_run_id):

    mlflow.pytorch.log_model(
        model,
        name="model",
        serialization_format="pickle"
    )

    print("Model logged successfully.")



# Register Model

model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="PetBreedClassifier"
)

print("\nModel registered successfully!")
print("Model name:", registered_model.name)
print("Model version:", registered_model.version)