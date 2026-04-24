# 📘 Fraud Detection – Technical Documentation

## 1. Introduction

This project implements a scalable machine learning pipeline for detecting fraudulent credit card transactions.
It transforms a notebook-based workflow into a **modular, production-style system**.

---

## 2. System Design

The system follows a pipeline architecture:

1. Data ingestion
2. Feature engineering
3. Preprocessing
4. Model training
5. Evaluation
6. Artifact persistence

---

## 3. Data Pipeline

### 3.1 Data Loading

Handled in:

```
src/data/load_data.py
```

* Reads CSV files using pandas
* Parses datetime columns:

  * `trans_date_trans_time`
  * `dob`

---

### 3.2 Feature Engineering

Implemented in:

```
src/features/build_features.py
```

#### Time-Based Features

* Month
* Day
* Part of day (morning / afternoon / evening / night)

#### Behavioral Feature

* Time since last transaction (per credit card)

#### Geospatial Feature

* Distance between customer and merchant using Haversine formula

---

### 3.3 Preprocessing

Implemented in:

```
src/data/preprocess.py
```

Steps:

* Remove personally identifiable information
* Drop transaction identifiers
* Convert DOB → Age
* Remove raw datetime columns
* Remove coordinate columns after transformation

---

## 4. Modeling

### 4.1 CatBoost

* Native handling of categorical features
* Optimized for recall
* Uses class weighting for imbalance

### 4.2 XGBoost

* Requires preprocessing pipeline:

  * Target Encoding (high-cardinality)
  * One-Hot Encoding (low-cardinality)
* Uses `scale_pos_weight` for imbalance handling

---

## 5. Training Pipeline

Defined in:

```
src/pipelines/training_pipeline.py
```

### Steps:

1. Load configuration
2. Load dataset
3. Apply feature engineering
4. Preprocess data
5. Train-test split (stratified)
6. Model selection
7. Training
8. Evaluation
9. Save artifacts

---

## 6. Evaluation

Implemented in:

```
src/models/evaluate.py
```

Metrics used:

* Recall (primary)
* F1 Score
* Classification Report

---

## 7. Configuration Management

Configuration file:

```
configs/config.yaml
```

Controls:

* Data paths
* Feature selection
* Model parameters
* Training settings
* Output paths

---

## 8. Outputs

Artifacts generated:

### Models

```
outputs/models/
```

### Metrics

```
outputs/metrics/
```

---

## 9. Reproducibility

Ensured via:

* Fixed random seeds
* Config-driven pipeline
* Deterministic preprocessing steps

---

## 10. Scalability Considerations

The system is designed for extension:

* Easy addition of new models
* Config-based experimentation
* Separation of feature engineering and modeling
* Pipeline abstraction for automation

---

## 11. Future Enhancements

* Experiment tracking (MLflow)
* Hyperparameter tuning (Optuna)
* Model serving (FastAPI)
* CI/CD integration
* Docker support

---

## 12. Conclusion

This project demonstrates how to transition from a notebook-based workflow to a **production-ready machine learning system**, emphasizing modularity, reproducibility, and scalability.
