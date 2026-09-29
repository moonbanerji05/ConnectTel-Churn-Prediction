# ConnectTel Customer Churn Prediction

## Project Overview

ConnectTel Customer Churn Prediction is a machine learning project designed to identify customers who are likely to leave a telecom company.

The goal is to help ConnectTel identify high-risk customers early and take proactive retention actions.

## Objective

To build a reliable machine learning model that predicts customer churn and provides useful business insights.

## Project Workflow

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Data Preprocessing
- Logistic Regression
- Random Forest
- XGBoost
- 5-Fold Cross-Validation
- GridSearchCV Hyperparameter Tuning
- Model Evaluation
- SHAP Explainable AI
- Business Recommendations

## Feature Engineering

Three new features were created:

- `TotalChargesPerTenure`
- `ServiceCount`
- `HasStreaming`

## Machine Learning Models

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Tuned XGBoost

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Explainable AI

SHAP was used to identify the most important factors influencing customer churn predictions.

## Business Impact

The model can help ConnectTel:

- Identify high-risk customers
- Target month-to-month customers
- Focus on new customers
- Review high monthly-charge customers
- Create personalized retention strategies

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP

## Project Structure

```text
ConnectTel-Churn-Prediction/
├── ConnectTel_Churn_Prediction.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
├── requirements.txt
└── .gitignore

##Live Application
https://connecttel-churn-prediction-hne9hz67zr8jfkghjauxgj.streamlit.app/
