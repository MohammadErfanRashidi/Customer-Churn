# Customer Churn Prediction – Modeling Report

## Objective
Identify customers likely to churn using a logistic regression model.  
**Primary metric:** Recall – we want to catch as many actual churners as possible.

## Data Overview
- **Source:** `Customer-Churn_Processed.csv`
- **Samples:** 7,043 customers
- **Features:** 30 numerical features (already scaled/encoded)
- **Target:** `Churn_Num` (0 = stayed, 1 = churned)
- **Class Imbalance:** ~26.5% churn rate

## Modeling Approach

### 1. Baseline Logistic Regression
- Used `class_weight='balanced'` to account for class imbalance.
- Performed grid search over `C` and `penalty` with **recall** as the scoring metric.
- **Best parameters:** `C=0.1`, `penalty='l2'`, `solver='liblinear'`

**Baseline Test Results:**
| Class      | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| No Churn   | 0.90      | 0.70   | 0.79     |
| Churn      | 0.49      | 0.79   | 0.60     |

- **Recall of 79%** – Good at catching churners.
- **Precision of 49%** – About half of flagged customers are false positives.

### 2. Feature Engineering (Based on Coefficient Analysis)
Strongest predictors from baseline model:
| Feature                  | Coefficient | Interpretation                          |
|--------------------------|-------------|------------------------------------------|
| `Contract_Encoded`       | -0.73       | Longer contract → much lower churn risk  |
| `TenureGroup`            | -0.35       | Longer tenure → lower churn risk         |
| `HasFiberOptic`          | +0.32       | Fiber optic users more likely to churn   |
| `ChargeGroup`            | +0.27       | Higher charges → higher churn risk       |

**New features created:**
- `Contract_Tenure_Interaction` – captures combined effect of contract and tenure.
- `HighRisk_Flag` – fiber + month-to-month + high monthly charges.
- `Tenure_Charge_Ratio` – value-for-money stickiness.
- `Security_Tech_Support` – sum of online security and tech support flags.

### 3. Undersampling for Better Balance
- Applied `RandomUnderSampler` **only to training data**.
- Balanced training set to 1,049 churn / 1,049 non-churn samples.

**Undersampled Model Results:**
| Class      | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| No Churn   | 0.90      | 0.71   | 0.79     |
| Churn      | 0.49      | 0.80   | 0.61     |

- Recall improved slightly to **80%**, precision unchanged.

### 4. Threshold Tuning for Maximum Recall
We can adjust the decision threshold to capture more churners at the cost of lower precision.

| Threshold | Recall | Precision | Business Implication                         |
|-----------|--------|-----------|-----------------------------------------------|
| 0.5       | 0.80   | 0.49      | Default – balanced trade‑off                  |
| 0.3       | 0.95   | 0.38      | Catch 95% of churners; higher false positives |

**Chosen threshold:** `0.3` – maximizes recall while keeping false positives manageable for low‑cost retention campaigns.

## Final Model Details
- **Algorithm:** Logistic Regression
- **Hyperparameters:** `C=1`, `penalty='l2'`, `solver='liblinear'` (from undersampled grid search)
- **Training:** Undersampled to 50/50 class balance
- **Features:** Original 30 + 4 engineered features
- **Threshold:** 0.3 (applied after probability prediction)

## Key Findings
1. **Contract type and tenure are the dominant churn drivers.** Month‑to‑month customers with short tenure are highest risk.
2. **Fiber optic service correlates with higher churn** – may indicate pricing or quality issues worth investigating.
3. **Undersampling + feature engineering provided a modest recall boost (79% → 80%).**
4. **Lowering the threshold to 0.3 increased recall to 95%** – ideal for proactive retention outreach.
5. The model is simple, interpretable, and ready for deployment.