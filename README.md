# 🚀 Fraud Detection Pipeline using RAPIDS, cuML, and XGBoost (GPU)

A high-performance, GPU-accelerated machine learning pipeline for detecting fraudulent financial transactions using a real-world, large-scale dataset (6.3M+ records).

---

## 🧠 Overview
This project demonstrates how to design and deploy an end-to-end fraud detection system using:
- RAPIDS cuDF & cuML for GPU-accelerated data processing and ML
- XGBoost (GPU) for highly accurate, scalable classification
- Google Colab for experimentation

---

## 🛠️ Tech Stack

| Component        | Tool/Library            |
|------------------|-------------------------|
| Dataframe Engine | RAPIDS cuDF             |
| ML Models        | cuML SVM & Random Forest, XGBoost (GPU) |
| Evaluation       | scikit-learn            |
| Visualization    | matplotlib, seaborn     |
| Notebook Env     | Google Colab            |

---

## 🧪 Dataset Preparation

1. **Imbalanced Data Handling:**
   - Extracted all fraud cases (`isFraud == 1`)
   - Sampled non-fraud to build a 1M-row training set
   - Held out 50K rows for evaluation

2. **Data Cleaning & Transformation:**
   - Removed identifiers (`nameOrig`, `nameDest`)
   - One-hot encoded categorical features (e.g., `type`)
   - Scaled numeric features using cuML `StandardScaler`

---

## 📊 EDA Highlights

- Class imbalance visualization
- Log-scaled transaction amount distribution
- Transaction type breakdown
- Correlation heatmap
- Balance boxplots by class

---

## 🤖 Models Trained

| Model           | Precision | Recall | F1 Score | ROC AUC |
|----------------|-----------|--------|----------|---------|
| SVM (cuML)     | 19.5%     | 95.4%  | 32.4%    | N/A     |
| Random Forest  | **98.9%** | 74.0%  | 84.7%    | N/A     |
| XGBoost (GPU)  | 53.9%     | **99.7%** | 70.0% | **0.9995** |

---

## 🧩 Evaluation Pipeline

A reusable function:
- Accepts any dataset with `isFraud` column
- Loads trained XGBoost model and scaler (`joblib`)
- Scales features and makes predictions
- Prints:
  - Accuracy
  - Confusion Matrix

```python
def evaluate_model_on_dataset(dataset_path, model_path='xgb_model.joblib', scaler_path='scaler.joblib'):
    ...
```

---

## ✅ Results

- **Random Forest** → Extremely high precision, good for avoiding false alarms
- **XGBoost** → Exceptional recall and ROC AUC, ideal for catching all fraud cases

> Ideal model depends on business goals: maximize recall or reduce false positives

---

## 📂 Project Structure

```bash
├── notebooks/                  # Experiments and training
├── models/                     # Saved models and scalers (.joblib)
├── data/                       # Preprocessed datasets
├── utils/                      # Evaluation and helper functions
└── README.md                   # Project overview
```

---

## 📌 Lessons Learned

- GPU acceleration drastically speeds up large-scale ML workflows
- Class imbalance is critical in fraud detection and requires thoughtful sampling
- XGBoost with proper tuning (scale_pos_weight) performs best in real-world fraud tasks
- A clear pipeline makes experimentation, evaluation, and deployment scalable

---

## 👨‍💻 Author
**Ahmed Alhisan**  
Building scalable AI & analytics systems for real-world impact  
Connect on [LinkedIn](https://www.linkedin.com/) *(insert actual link)*

---

## ⭐ If you found this helpful
Please ⭐ star this repo or share it to support more GPU-based open-source fraud analytics!

---
