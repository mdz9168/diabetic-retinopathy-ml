# Diabetic Retinopathy Classification

This project applies machine learning to detect diabetic retinopathy (DR) based on clinical diagnostic features. DR is a diabetes complication that affects vision — early detection is key. Using the Debrecen dataset, we implemented multiple ML models to classify whether a patient has DR.

---

## 📊 Project Overview

- **Goal**: Predict the presence of diabetic retinopathy (binary classification: DR / No DR)
- **Dataset**: [UCI Debrecen Diabetic Retinopathy Dataset](https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set)
- **Features**: 19 numerical diagnostic values per patient
- **Tech Stack**: Python, scikit-learn, pandas, matplotlib, seaborn
- **Models**:
  - Logistic Regression (baseline)
  - AdaBoost (GridSearchCV tuning)
  - Random Forest (30 estimators)

---

## 🧠 Key Learnings

- **AdaBoost** showed the highest accuracy (~70%) after hyperparameter tuning.
- **Random Forest** performed well and supports feature importance.
- **Logistic Regression** was the most interpretable, but slightly less accurate.

---

## 🗂️ Project Structure

diabetic-retinopathy-ml/ ├── main.py # Pipeline script: runs all models ├── data/ │ └── Retinopathy_Debrecen.csv ├── src/ │ ├── preprocessing.py # Scaling, train-test split │ ├── models.py # Model definitions │ └── evaluation.py # Metrics + confusion matrix ├── outputs/ │ ├── figures/ # Confusion matrix plots │ └── reports/ # Classification reports (.json) ├── requirements.txt └── README.md


---

## 🚀 How to Run This Project

1. Clone this repo:
```bash
git clone https://github.com/mdz9168/diabetic-retinopathy-ml.git
cd diabetic-retinopathy-ml

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

python main.py

Model | Accuracy | Notes
---|---|---
Logistic Regression | ~65% | Linear baseline  
AdaBoost | ~70% | Tuned with GridSearchCV  
Random Forest | ~66% | 30 estimators, no tuning  

This project is licensed under the MIT License.  
See LICENSE for full text.


---

## 👤 About the Author

Hi! I'm **Moses Lian**, a data science & machine learning enthusiast.  
This project reflects my passion for applying ML in healthcare and real-world diagnostics.

- 🔗 [LinkedIn](https://www.linkedin.com/in/moses-lian/)
- 🐙 [GitHub](https://github.com/mdz9168)

