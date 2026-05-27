# 🌸 Classic ML Pipeline — Iris & Titanic

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-REST_API-black?logo=flask)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A production-style machine learning pipeline built on two classic datasets — **Iris** (multi-class classification) and **Titanic** (binary classification). Covers the full ML lifecycle: EDA → preprocessing → model training → evaluation → REST API deployment with Docker.

> **Purpose:** Demonstrate core ML fundamentals — data cleaning, feature engineering, sklearn pipelines, model comparison, and API deployment — in a clean, reproducible project structure.

---

## 📌 What This Project Covers

| Concept | Implementation |
|---|---|
| Exploratory Data Analysis | Matplotlib + Seaborn visualizations |
| Data Preprocessing | Imputation, encoding, scaling via sklearn Pipeline |
| Model Training | Logistic Regression, Random Forest, XGBoost |
| Model Evaluation | Accuracy, F1, confusion matrix, classification report |
| Model Persistence | joblib serialization |
| REST API | Flask with `/predict/iris` and `/predict/titanic` endpoints |
| Containerization | Dockerfile + docker-compose |
| Testing | pytest with Flask test client |

---

## 🗂️ Project Structure

```
classic-ml-pipeline/
├── data/
│   ├── raw/                  # Original CSVs
│   └── processed/            # Cleaned data
├── notebooks/
│   ├── 01_iris_eda.ipynb     # Iris EDA and analysis
│   └── 02_titanic_eda.ipynb  # Titanic EDA and analysis
├── src/
│   ├── preprocess.py         # Cleaning + feature engineering
│   ├── train.py              # Model training + evaluation
│   └── predict.py            # Inference utilities
├── models/                   # Saved .pkl model files
├── api/
│   └── app.py                # Flask REST API
├── tests/
│   ├── test_preprocess.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/classic-ml-pipeline.git
cd classic-ml-pipeline
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the models

```bash
cd src
python train.py
```

### 4. Run the API locally

```bash
python api/app.py
```

### 5. Run with Docker

```bash
docker-compose up --build
```

---

## 🚀 API Usage

### Health check

```bash
curl http://localhost:5000/health
# → {"status": "ok"}
```

### Iris prediction

```bash
curl -X POST http://localhost:5000/predict/iris \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**Response:**
```json
{
  "prediction": "setosa",
  "confidence": 0.97
}
```

### Titanic prediction

```bash
curl -X POST http://localhost:5000/predict/titanic \
  -H "Content-Type: application/json" \
  -d '{"pclass": 1, "sex": 1, "age": 29, "sibsp": 0, "parch": 0, "fare": 211, "embarked": 1}'
```

**Response:**
```json
{
  "survived": true,
  "survival_probability": 0.91
}
```

---

## 📊 Model Results

### Iris Dataset
| Model | Accuracy |
|---|---|
| Logistic Regression | ~97% |
| Random Forest | ~98% |

### Titanic Dataset
| Model | Accuracy |
|---|---|
| Logistic Regression | ~80% |
| Random Forest | ~83% |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **ML:** Scikit-learn, XGBoost
- **API:** Flask
- **Serialization:** joblib
- **Containerization:** Docker, Docker Compose
- **Testing:** pytest
- **Visualization:** Matplotlib, Seaborn

---

## 📚 Key Learnings

- Building end-to-end sklearn `Pipeline` objects that bundle preprocessing + model
- Handling missing data, categorical encoding, and feature scaling correctly
- Comparing multiple classifiers systematically with proper evaluation metrics
- Serving a trained model as a REST API with Flask
- Containerizing a Python ML app with Docker

---

## 🗺️ Part of ML Learning Roadmap

This is **Project 1 of 10** in a progressive ML + GenAI portfolio:

| # | Project | Skills |
|---|---|---|
| ✅ 1 | Classic ML Pipeline (this project) | EDA, Sklearn, Flask, Docker |
| 2 | House Price Predictor | Regression, Feature Eng., Streamlit |
| 3 | Churn Classifier | XGBoost, SHAP, FastAPI |
| 4 | Image Classifier | PyTorch, CNN, MLflow |
| 5 | Sentiment Analyzer | HuggingFace, BERT, NLP |
| ... | ... | ... |

---

## 👤 Author

**Your Name**
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

