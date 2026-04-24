import os
import json
import yaml
import joblib
from datetime import datetime


# =========================
# CONFIG LOADER
# =========================
def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# =========================
# DIRECTORY HANDLER
# =========================
def create_dir(path: str):
    """
    Create directory if it doesn't exist
    """
    if not os.path.exists(path):
        os.makedirs(path)


# =========================
# TIMESTAMP GENERATOR
# =========================
def get_timestamp() -> str:
    """
    Returns formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================
# SAVE MODEL
# =========================
def save_model(model, output_dir: str, model_name: str):
    """
    Save trained model using joblib
    """
    create_dir(output_dir)

    timestamp = get_timestamp()
    file_path = os.path.join(output_dir, f"{model_name}_{timestamp}.pkl")

    joblib.dump(model, file_path)
    print(f"Model saved at: {file_path}")


# =========================
# SAVE METRICS
# =========================
def save_metrics(metrics: dict, output_dir: str, filename: str = "metrics.json"):
    """
    Save evaluation metrics as JSON
    """
    create_dir(output_dir)

    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved at: {file_path}")


# =========================
# LOGGING HELPER
# =========================
def log(message: str):
    """
    Simple logger with timestamp
    """
    print(f"[{get_timestamp()}] {message}")