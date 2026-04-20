import pandas as pd

# Hardcoded median of MonthlyCharges_Scaled from training data (used in modeling notebook)
MEDIAN_MONTHLY_CHARGES_SCALED = -0.168120


def add_engineered_features(df):
    """
    Takes the 32-column preprocessed DataFrame (without target)
    Returns DataFrame with 4 additional engineered features (total 36 columns)
    """
    df = df.copy()
    
    # 1. Contract_Tenure_Interaction
    df['Contract_Tenure_Interaction'] = df['Contract_Encoded'] * df['tenure_Scaled']
    
    # 2. HighRisk_Flag: Fiber optic + Month-to-month + high monthly charges
    df['HighRisk_Flag'] = (
        (df['HasFiberOptic'] == 1) & 
        (df['Contract_Encoded'] == 0) & 
        (df['MonthlyCharges_Scaled'] > MEDIAN_MONTHLY_CHARGES_SCALED)
    ).astype(int)
    
    # 3. Tenure_Charge_Ratio
    # Add small constant to avoid division by zero (as in notebook)
    df['Tenure_Charge_Ratio'] = df['tenure_Scaled'] / (df['MonthlyCharges_Scaled'] + 0.001)
    
    # 4. Security_Tech_Support
    df['Security_Tech_Support'] = df['HasOnlineSecurity'] + df['HasTechSupport']
    
    return df