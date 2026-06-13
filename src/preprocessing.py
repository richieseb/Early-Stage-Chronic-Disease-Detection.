import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_and_scale_data(df):
    """
    Handles missing values via median/mode imputation and maps features
    to a standardized [0, 1] scale.
    """
    processed_df = df.copy()
    
    # Impute missing continuous features using the training median
    for col in ['Blood_Pressure', 'Serum_Creatinine']:
        if processed_df[col].isnull().any():
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            
    # Separate independent feature attributes and label matrices
    X = processed_df[['Blood_Pressure', 'Serum_Creatinine']].values
    y = processed_df['Diagnosis_Target'].values
    
    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler
