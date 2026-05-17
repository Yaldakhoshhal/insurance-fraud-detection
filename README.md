# Insurance Fraud Detection

Machine learning project for detecting fraudulent insurance claims using classification models, imbalance handling techniques, threshold tuning, and hyperparameter optimization.

---

## Project Overview

This project explores multiple machine learning approaches for insurance fraud detection, including:

- Logistic Regression
- Random Forest
- Gradient Boosting

The workflow includes:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Handling class imbalance with SMOTE
- Threshold optimization
- Hyperparameter tuning with GridSearchCV and Optuna
- Model evaluation using Precision, Recall, F1-Score, and ROC-AUC

---

## Final Model Performance

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.132 | 0.816 | 0.228 | 0.807 |
| Random Forest | 0.189 | 0.378 | 0.252 | 0.818 |
| Gradient Boosting | 0.208 | 0.438 | 0.282 | 0.835 |

The optimized Gradient Boosting model achieved the strongest overall performance and provided the best balance between fraud detection capability and classification reliability.

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- Optuna
- matplotlib
- seaborn

---

## Repository Structure

```text
insurance-fraud-detection/
│
├── data/
├── insurance_fraud_detection.ipynb
└── README.md
```

---

## Business Insight

Fraud detection is a highly imbalanced classification problem where improving fraud detection capability often increases false positives. This project highlights the importance of combining model optimization with threshold tuning to build more practical fraud detection systems for real-world insurance applications.