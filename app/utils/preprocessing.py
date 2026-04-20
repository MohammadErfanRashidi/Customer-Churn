import pandas as pd
import json
import os

# Load parameters from JSON files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'scaler_params.json'), 'r') as f:
    SCALER_PARAMS = json.load(f)

with open(os.path.join(BASE_DIR, 'encoding_mappings.json'), 'r') as f:
    ENCODING_MAPPINGS = json.load(f)


def preprocess_input(df):
    """
    Takes a DataFrame with original cleaned columns (19 features, no customerID, no target)
    Returns a DataFrame with 32 base features (same structure as Customer-Churn_Processed.csv)
    """
    df = df.copy()
    
    # Drop duplicates if any (consistent with training)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 1. Binary mapping
    binary_map = {'Yes': 1, 'No': 0}
    df['Partner_Num'] = df['Partner'].map(binary_map)
    df['Dependents_Num'] = df['Dependents'].map(binary_map)
    df['PhoneService_Num'] = df['PhoneService'].map(binary_map)
    df['PaperlessBilling_Num'] = df['PaperlessBilling'].map(binary_map)
    df['Gender_Num'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # Drop original binary columns
    df.drop(columns=['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'gender'], inplace=True)
    
    # 2. Label encode multi-cat columns using saved mappings
    for col, categories in ENCODING_MAPPINGS.items():
        # Create mapping dict from category to index
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        df[f'{col}_Encoded'] = df[col].map(mapping)
        # Handle unseen categories by assigning -1 (or most frequent) - but here we assume valid input
        if df[f'{col}_Encoded'].isnull().any():
            # Fallback to most frequent (first category)
            df[f'{col}_Encoded'].fillna(0, inplace=True)
    
    # Drop original multi-cat columns
    df.drop(columns=list(ENCODING_MAPPINGS.keys()), inplace=True)
    
    # 3. Feature engineering
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 100], labels=[0,1,2,3]).astype(int)
    df['ChargeGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 100, 150], labels=[0,1,2,3]).astype(int)
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    service_cols = ['OnlineSecurity_Encoded', 'OnlineBackup_Encoded', 'DeviceProtection_Encoded',
                    'TechSupport_Encoded', 'StreamingTV_Encoded', 'StreamingMovies_Encoded']
    df['ServiceCount'] = df[service_cols].sum(axis=1)
    
    df['HasMultipleLines'] = (df['MultipleLines_Encoded'] == 2).astype(int)
    df['HasInternet'] = (df['InternetService_Encoded'] != 2).astype(int)
    df['HasFiberOptic'] = (df['InternetService_Encoded'] == 1).astype(int)
    df['HasDSL'] = (df['InternetService_Encoded'] == 0).astype(int)
    df['HasOnlineSecurity'] = (df['OnlineSecurity_Encoded'] == 2).astype(int)
    df['HasTechSupport'] = (df['TechSupport_Encoded'] == 2).astype(int)
    df['TotalServices'] = df['HasInternet'] + df['PhoneService_Num'] + df['ServiceCount']
    df['HasPartnerOrDependents'] = ((df['Partner_Num'] == 1) | (df['Dependents_Num'] == 1)).astype(int)
    
    # 4. Apply standard scaling
    for col, params in SCALER_PARAMS.items():
        mean = params['mean']
        std = params['std']
        df[f'{col}_Scaled'] = (df[col] - mean) / std
    
    # 5. Drop intermediate columns that won't be used
    cols_to_drop = ['tenure', 'MonthlyCharges', 'ServiceCount', 'AvgMonthlySpend', 'TotalCharges']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # 6. Ensure column order matches training (exclude Churn_Num which is target)
    final_columns = [
        'SeniorCitizen', 'Partner_Num', 'Dependents_Num', 'PhoneService_Num',
        'PaperlessBilling_Num', 'Gender_Num', 'MultipleLines_Encoded',
        'InternetService_Encoded', 'OnlineSecurity_Encoded', 'OnlineBackup_Encoded',
        'DeviceProtection_Encoded', 'TechSupport_Encoded', 'StreamingTV_Encoded',
        'StreamingMovies_Encoded', 'Contract_Encoded', 'PaymentMethod_Encoded',
        'TenureGroup', 'ChargeGroup', 'HasMultipleLines', 'HasInternet',
        'HasFiberOptic', 'HasDSL', 'HasOnlineSecurity', 'HasTechSupport',
        'TotalServices', 'HasPartnerOrDependents', 'tenure_Scaled',
        'MonthlyCharges_Scaled', 'ServiceCount_Scaled', 'AvgMonthlySpend_Scaled',
        'TotalCharges_Scaled'
    ]
    
    # Reorder to match exactly
    df = df[final_columns]
    
    return df