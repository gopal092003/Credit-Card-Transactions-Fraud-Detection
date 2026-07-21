# Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions. The project follows a modular architecture with separate components for data processing, feature engineering, model training, and evaluation, making it easy to maintain, experiment with, and extend.

---

## Overview

Credit card fraud detection is a highly imbalanced classification problem where correctly identifying fraudulent transactions is more important than maximizing overall accuracy.

This project focuses on building a reproducible machine learning workflow that processes transaction data, engineers meaningful features, trains classification models, and evaluates their performance using metrics suited for imbalanced datasets.

---

## Features

- Modular machine learning pipeline
- Configuration-driven experiments using YAML
- Automated feature engineering
- Support for CatBoost and XGBoost
- Model evaluation using Recall and F1 Score
- Automatic model and metrics saving
- Organized project structure for scalability

---

## Pipeline

The training workflow consists of the following stages:

1. Load transaction data
2. Perform feature engineering
3. Preprocess the dataset
4. Train the selected model
5. Evaluate model performance
6. Save the trained model and evaluation metrics

> Add the pipeline diagram below once available.

```text
assets/pipeline.png
```

---

## Project Structure

```text
fraud-detection/
│
├── configs/
│   └── config.yaml
│
├── data/
├── notebooks/
├── outputs/
│
├── assets/
│   └── pipeline.png
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   └── utils/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Feature Engineering

The pipeline creates several features to improve fraud detection, including:

- Transaction time features
- Time elapsed since the previous transaction
- Geographic distance between transactions using the Haversine formula
- Customer age
- Additional preprocessing required by the selected model

---

## Models

The project currently supports:

### CatBoost

- Native handling of categorical features
- Minimal preprocessing
- Suitable for structured tabular datasets

### XGBoost

- Gradient boosting with encoded categorical features
- Flexible and widely used for classification tasks

The model can be selected through the configuration file without modifying the training pipeline.

---

## Dataset

The dataset is available on Kaggle:

https://www.kaggle.com/datasets/kartik2112/fraud-detection

After downloading, place the files in the following directory:

```text
data/raw/
├── fraudTrain.csv
└── fraudTest.csv
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/gopal092003/fraud-detection.git

cd fraud-detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Start the training pipeline with:

```bash
python main.py
```

---

## Configuration

All experiment settings are managed through:

```text
configs/config.yaml
```

You can configure:

- Model selection
- Hyperparameters
- Feature engineering options
- Training settings

This makes it easy to reproduce experiments and compare different configurations.

---

## Outputs

After training, the pipeline automatically generates:

```text
outputs/
├── models/
│   └── trained_model.pkl
│
└── metrics/
    └── evaluation.json
```

---

## Evaluation

Because fraudulent transactions represent only a small fraction of the dataset, the project prioritizes metrics that better reflect fraud detection performance.

Current evaluation metrics include:

- Recall
- F1 Score

---

## Future Improvements

Some planned enhancements include:

- MLflow experiment tracking
- Hyperparameter optimization with Optuna
- FastAPI inference service
- Docker support
- Model versioning
- Automated data validation

---

## Author

**Gopal Gupta**

GitHub: https://github.com/gopal092003

---

## License

This project is licensed under the MIT License.
