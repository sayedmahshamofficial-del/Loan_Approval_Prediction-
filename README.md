# 🏦 Loan Approval Prediction System

A machine learning project that predicts whether a loan application is likely to be approved or rejected based on applicant and financial information.

The project includes data cleaning, preprocessing, model comparison, automatic best-model selection, evaluation, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Loan approval is an important decision in the banking and financial sector.

This project uses machine learning classification algorithms to analyze applicant information and predict:

- Approved
- Rejected

The system provides a web-based interface where users can enter applicant information and receive a prediction with probability scores.

---

## 🎯 Objectives

- Clean and prepare the loan approval dataset.
- Handle missing and categorical data.
- Perform feature preprocessing.
- Train multiple machine learning models.
- Compare model performance.
- Automatically select the best-performing model.
- Save the trained model.
- Build an interactive Streamlit dashboard.
- Visualize model performance.

---

## 🧠 Machine Learning Models

The following classification algorithms are evaluated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The final model is selected automatically based on the highest F1 Score.

---

## 📊 Dataset

The dataset contains information about loan applicants.

### Features

| Feature | Description |
|---|---|
| Gender | Applicant gender |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Education level |
| Employment_Status | Employment status |
| Applicant_Income | Applicant income |
| Coapplicant_Income | Coapplicant income |
| Loan_Amount | Requested loan amount |
| Loan_Term | Loan repayment period |
| Credit_History | Credit history indicator |
| Property_Area | Property location |
| Age | Applicant age |

### Target

```text
Loan_Status