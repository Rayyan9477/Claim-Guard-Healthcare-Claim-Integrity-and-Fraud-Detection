import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_training_data(data):
    # For training data with claim_status
    data = data.dropna()
    data['cpt_code'] = data['cpt_code'].astype(str)
    cpt_dummies = pd.get_dummies(data['cpt_code'], prefix='cpt')
    X = pd.concat([data[['claim_amount']], cpt_dummies], axis=1)
    y = (data['claim_status'] == 'denied').astype(int)
    return X, y

def preprocess_prediction_data(data):
    # For prediction data without claim_status
    data['cpt_code'] = data['cpt_code'].astype(str)
    cpt_dummies = pd.get_dummies(data['cpt_code'], prefix='cpt')
    X = pd.concat([data[['claim_amount']], cpt_dummies], axis=1)
    return X

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def save_model(model, model_path, feature_names=None):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': model, 'feature_names': feature_names}, model_path)

def load_model(model_path):
    data = joblib.load(model_path)
    return data['model'], data['feature_names']

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'dataset.csv')
    model_path = os.path.join(os.path.dirname(current_dir), 'models', 'model.pkl')
    
    data = load_data(data_path)
    X, y = preprocess_training_data(data)
    model = train_model(X, y)
    save_model(model, model_path, X.columns.tolist())
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()