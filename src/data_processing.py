import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Fill missing values
    data = data.fillna(method='ffill')
    
    # Convert data types if necessary
    # Example: data['claim_amount'] = data['claim_amount'].astype(float)
    
    return data

def prepare_data(data):
    # Feature selection and engineering
    features = data[['claim_amount', 'cpt_code', 'claim_status']]
    labels = data['claim_denial']  # Assuming 'claim_denial' is the target variable
    return features, labels

def preprocess_data(data):
    # Convert categorical variables to numerical
    data['claim_status'] = data['claim_status'].map({'approved': 1, 'denied': 0})
    return data