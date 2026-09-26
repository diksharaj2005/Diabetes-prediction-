# 🩺 Diabetes Risk Predictor
### Lab Assignment 01 — Predicting Diabetes with Multilayer Perceptron
**CSE 4192 — Machine Learning Projects with Python**

---

## 📌 Overview
A Streamlit web application that predicts whether a patient is likely to have diabetes using a trained **Multilayer Perceptron (MLP)** model. Built on the **Pima Indians Diabetes Dataset**.

---

## 👥 Team Members
| Name | Reg. No. |
|------|----------|
| Umang Kumar Chourasia | |
| Aditya Kumar Sahay | |
| Khushi Singh | |
| Diksha Raj | |

---

## 🗂️ Project Structure
```
diabetes_app/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── diabetes_mlp.keras      # Trained MLP model
├── diabetes_scaler.pkl     # Fitted StandardScaler
├── feature_columns.pkl     # Training feature column names
└── README.md
```

---

## 🧠 Model Architecture
```
Input Features (13)
       ↓
Dense(32, ReLU) → Dropout(0.2)
       ↓
Dense(16, ReLU) → Dropout(0.2)
       ↓
Dense(1, Sigmoid)
       ↓
Diabetes Prediction
```
- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Dataset:** Pima Indians Diabetes Dataset (768 samples, 8 features)

---

## ⚙️ Features
- 🔍 Real-time diabetes risk prediction
- 📊 Probability score with visual progress bar
- 🚩 Key risk factor identification
- 🧪 Built-in deployment test cases (Table 10)
- 🚫 Invalid input validation — app never crashes

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/diabetes-mlp-app.git
cd diabetes-mlp-app

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📦 Dependencies
- `streamlit`
- `tensorflow`
- `scikit-learn`
- `pandas`
- `numpy`
- `joblib`

---

## 📊 Dataset
**Pima Indians Diabetes Database**
- Source: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 observations, 8 features
- Binary classification: `0` → Non-Diabetic, `1` → Diabetic

---

## ⚠️ Disclaimer
This application is developed for **academic purposes only** as part of a university lab assignment. It is **not** intended to be used as a medical diagnostic tool. Always consult a qualified healthcare professional for medical advice.
