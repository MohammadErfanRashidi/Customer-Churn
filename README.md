# Customer Churn Prediction

A complete machine learning pipeline that predicts telecom customer churn, from raw data cleaning and feature engineering to model deployment via an interactive Streamlit web application.

**Live Demo:** [https://customer-churn-muqebcivtagcv24jhlczm9.streamlit.app/]

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Data Source](#data-source)
- [Methodology](#methodology)
  - [Data Cleaning](#data-cleaning)
  - [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
  - [Modeling](#modeling)
- [Key Findings](#key-findings)
- [Web Application](#web-application)
- [Repository Structure](#repository-structure)
- [Installation & Local Usage](#installation--local-usage)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

## Project Overview

This project demonstrates a full data science workflow:

1.  Cleaning and validating raw customer data.
2.  Engineering predictive features from categorical and numerical inputs.
3.  Training a logistic regression model with class imbalance handling.
4.  Packaging the preprocessing and model into a reusable pipeline.
5.  Deploying the solution as a user-friendly web app for both single predictions and batch processing.

The entire process is documented in Jupyter notebooks, and the final model is served through a Streamlit application that anyone can use without writing code.

## Business Problem

Customer churn (attrition) is a major cost driver for subscription-based businesses. Identifying customers who are likely to leave allows the company to proactively offer retention incentives, ultimately reducing revenue loss.

The goal of this project is to build a predictive model that:

- Flags at-risk customers with high recall (catching as many potential churners as possible).
- Provides actionable probabilities rather than just binary labels.
- Can be easily used by non-technical stakeholders via a simple interface.

## Data Source

The dataset used is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), which contains information about a fictional telecom company's customers.

- **Rows:** 7,043
- **Features:** Demographics, account information, services subscribed, and billing details.
- **Target:** `Churn` (Yes/No).

## Methodology

All steps are reproducible via the Jupyter notebooks included in the `/notebooks` directory.

### Data Cleaning

- Dropped the unique identifier `customerID` as it holds no predictive power.
- Converted `TotalCharges` from string to numeric, handling blank strings by coercing to zero (these represented new customers with no prior charges).
- Verified no remaining missing values.
- Removed duplicate rows.

### Preprocessing & Feature Engineering

The preprocessing pipeline (`utils/preprocessing.py`) replicates the exact transformations from the notebook:

- **Binary Encoding:** Converted Yes/No columns (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `gender`) to numeric indicators.
- **Label Encoding:** Applied `pandas.factorize` to multi-category columns such as `Contract`, `InternetService`, and `PaymentMethod`. The category mappings are saved in `encoding_mappings.json` for consistency.
- **Custom Features:**
  - `TenureGroup`: Binned tenure into four groups (0-12, 13-24, 25-48, 49+ months).
  - `ChargeGroup`: Binned monthly charges into four tiers.
  - `AvgMonthlySpend`: Total charges divided by (tenure + 1).
  - `ServiceCount`: Count of additional services subscribed.
  - `HasPartnerOrDependents`: Combined indicator for household stability.
  - Several binary flags derived from service columns (e.g., `HasFiberOptic`, `HasOnlineSecurity`).
- **Scaling:** Standardized numerical features (`tenure`, `MonthlyCharges`, `ServiceCount`, `AvgMonthlySpend`, `TotalCharges`) using parameters saved in `scaler_params.json`.

The final base feature set contains 32 columns.

### Modeling

The modeling notebook (`notebooks/modeling.ipynb`) details the training process:

- **Train/Test Split:** 80/20 stratified split to preserve class distribution.
- **Class Imbalance:** The dataset has a 26.5% churn rate. To improve recall on the minority class, `RandomUnderSampler` was applied to the training data only.
- **Additional Engineering:** Four more features were created immediately before training to capture interactions and risk profiles:
  - `Contract_Tenure_Interaction`
  - `HighRisk_Flag` (Fiber optic + Month-to-month contract + high charges)
  - `Tenure_Charge_Ratio`
  - `Security_Tech_Support` (sum of online security and tech support indicators)
- **Model Selection:** Logistic Regression with grid search over regularization strength (`C`), penalty (`l1`, `l2`), and solver. The scoring metric was **recall**.
- **Best Model:** `C=0.01`, `penalty='l1'`, `solver='liblinear'`.

**Model Performance on Holdout Test Set (Threshold = 0.5):**

| Metric    | Value |
|-----------|-------|
| Recall    | 0.80  |
| Precision | 0.49  |
| F1-Score  | 0.61  |

*The threshold can be adjusted in the app to balance catching more churners (higher recall) against reducing false positives (higher precision).*

## Key Findings

Analysis of the model coefficients and feature importance revealed several strong predictors of churn:

1.  **Contract Type:** Month-to-month contracts are the single largest risk factor for churn.
2.  **Tenure:** Customers with shorter tenure (less than 12 months) exhibit significantly higher churn rates.
3.  **Fiber Optic Service:** Customers with fiber optic internet, especially on month-to-month plans, have elevated churn risk.
4.  **Lack of Support Services:** Absence of online security and tech support correlates with higher churn probability.
5.  **Payment Method:** Customers paying via electronic check are more likely to churn compared to those using automatic bank transfers or credit cards.

These insights suggest that retention efforts should focus on new customers, those on flexible contracts, and encouraging adoption of security/support add-ons.

## Web Application

The trained model is served via a Streamlit application located in the `/app` directory.

**Features:**

- **Single Customer Prediction:** Fill in a form with all required customer attributes and receive an instant churn probability. A slider allows adjusting the classification threshold to see how the prediction changes.
- **Batch CSV Upload:** Upload a CSV file containing multiple customer records. The app processes the file, appends churn probabilities and predictions, and provides a download link for the results.
- **Clean, Responsive UI:** The interface is organized into logical sections for ease of use.

## Repository Structure

├── app/
│ ├── app.py # Streamlit application entry point
│ ├──requirements.txt # Python dependencies
│ └── model/
│ └── churn_logreg_model.pkl # Serialized trained model
├── utils/
│ ├── preprocessing.py # Reusable data preprocessing pipeline
│ ├── feature_engineering.py # Additional feature creation for model input
│ ├── scaler_params.json # Scaling parameters from training
│ └── encoding_mappings.json # Label encoding category mappings
├── notebooks/
│ ├── data_cleaning.ipynb # Raw data inspection and cleaning
│ ├── preprocessing.ipynb # Feature engineering and encoding
│ └── modeling.ipynb # Model training, evaluation, and feature analysis
├── .gitignore
└── README.md

## Technologies Used

- **Python 3.13**
- **Streamlit** (web application framework)
- **Pandas & NumPy** (data manipulation)
- **Scikit-learn** (preprocessing, modeling, evaluation)
- **Imbalanced-learn** (undersampling for class imbalance)
- **Jupyter Notebook** (exploratory analysis and prototyping)
- **Git & GitHub** (version control and collaboration)
