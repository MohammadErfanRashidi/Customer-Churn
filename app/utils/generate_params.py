import pandas as pd
import json

# Load cleaned data
url = 'https://raw.githubusercontent.com/MohammadErfanRashidi/Customer-Churn/refs/heads/main/data/Customer-Churn_Cleaned.csv'
df = pd.read_csv(url)

# Drop duplicates (as in preprocessing)
df = df.drop_duplicates().reset_index(drop=True)

# Binary mapping
yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in yes_no_cols:
    df[f'{col}_Num'] = df[col].map({'Yes': 1, 'No': 0})
df['Gender_Num'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Churn_Num'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop original binary/categorical columns that will be replaced
df = df.drop(columns=yes_no_cols + ['gender', 'Churn'])

# Multi-category columns to encode
multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

# Save encoding mappings (categories in order of appearance)
encoding_mappings = {}
for col in multi_cat_cols:
    # pd.factorize returns codes and uniques
    codes, uniques = pd.factorize(df[col])
    encoding_mappings[col] = uniques.tolist()
    # Add encoded column to dataframe for further steps
    df[f'{col}_Encoded'] = codes

# Drop original multi-cat columns
df = df.drop(columns=multi_cat_cols)

# Feature engineering (to compute scaling parameters)
df['TenureGroup'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 100], labels=[0,1,2,3]).astype(int)
df['ChargeGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 100, 150], labels=[0,1,2,3]).astype(int)
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)

service_cols = ['OnlineSecurity_Encoded', 'OnlineBackup_Encoded', 'DeviceProtection_Encoded',
                'TechSupport_Encoded', 'StreamingTV_Encoded', 'StreamingMovies_Encoded']
df['ServiceCount'] = df[service_cols].sum(axis=1)

# Binary indicators
df['HasMultipleLines'] = (df['MultipleLines_Encoded'] == 2).astype(int)
df['HasInternet'] = (df['InternetService_Encoded'] != 2).astype(int)
df['HasFiberOptic'] = (df['InternetService_Encoded'] == 1).astype(int)
df['HasDSL'] = (df['InternetService_Encoded'] == 0).astype(int)
df['HasOnlineSecurity'] = (df['OnlineSecurity_Encoded'] == 2).astype(int)
df['HasTechSupport'] = (df['TechSupport_Encoded'] == 2).astype(int)
df['TotalServices'] = df['HasInternet'] + df['PhoneService_Num'] + df['ServiceCount']
df['HasPartnerOrDependents'] = ((df['Partner_Num'] == 1) | (df['Dependents_Num'] == 1)).astype(int)

# Scaling parameters
scale_cols = ['tenure', 'MonthlyCharges', 'ServiceCount', 'AvgMonthlySpend', 'TotalCharges']
scaler_params = {}
for col in scale_cols:
    scaler_params[col] = {'mean': df[col].mean(), 'std': df[col].std()}

# Save to JSON files
with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=4)

with open('encoding_mappings.json', 'w') as f:
    json.dump(encoding_mappings, f, indent=4)

print("✅ Saved scaler_params.json and encoding_mappings.json")