# 💳 Credit Card Fraud Detection

## 📌 Overview

This project detects fraudulent credit card transactions using machine learning.
It is designed as a **modular, scalable ML pipeline** with proper project structure, configuration management, and reproducible workflows.

---

## 🧠 Pipeline Overview

![ML Pipeline](assets/pipeline.png)

The pipeline includes:

* Data loading
* Feature engineering
* Preprocessing
* Model training (CatBoost / XGBoost)
* Evaluation (Recall, F1-score)
* Model & metrics saving

---

## 🏗️ Project Structure

```
fraud-detection/
│
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   └── utils/
│
├── configs/
│   └── config.yaml
│
├── outputs/
├── assets/
│   └── pipeline.png
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Key Features

* 📦 **Modular pipeline-based architecture**
* ⚙️ **Config-driven experimentation**
* 🧠 **Feature engineering**

  * Time-based features
  * Time since last transaction
  * Distance using Haversine formula
  * Age calculation
* 🤖 **Model support**

  * CatBoost (native categorical handling)
  * XGBoost (encoding pipeline)
* 📊 **Evaluation metrics**

  * Recall (priority)
  * F1 Score

---

## 📂 Dataset

Download from Kaggle:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Place files in:

```
data/raw/
  ├── fraudTrain.csv
  └── fraudTest.csv
```

---

## 🚀 How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run pipeline

```
python main.py
```

---

## ⚙️ Configuration

All parameters are controlled via:

```
configs/config.yaml
```

You can:

* Switch models (CatBoost ↔ XGBoost)
* Adjust hyperparameters
* Modify feature settings

---

## 📊 Output

Generated automatically:

```
outputs/
├── models/      # saved models (.pkl)
└── metrics/     # evaluation results (.json)
```

---

## 🎯 Key Insights

* Fraud detection is **highly imbalanced** → Recall is critical
* Feature engineering significantly improves performance
* CatBoost handles categorical data effectively without encoding
* XGBoost performs well with proper preprocessing

---

## 🔮 Future Improvements

* MLflow experiment tracking
* Hyperparameter tuning (Optuna)
* Real-time inference API (FastAPI)
* Dockerization

---

## 👨‍💻 Author

github.com/gopal0992003
