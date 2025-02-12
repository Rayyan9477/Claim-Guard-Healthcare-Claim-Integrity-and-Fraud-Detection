import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_training_data(data):
    # Drop rows with missing values
    data = data.dropna()

    # Define categorical and numerical features
    categorical_features = ['gender', 'insurance_provider', 'place_of_service', 'billing_code', 'diagnosis_code']
    numerical_features = ['patient_age', 'claim_amount', 'submitted_charges', 'allowed_amount', 'copay_amount', 'deductible_amount']

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),  # Keep numerical features as is
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # One-hot encode categorical features
        ],
        remainder='passthrough'  # Drop other columns
    )

    # Prepare features (X) and target (y)
    X = data.drop(['claim_id', 'claim_status', 'denial_reason', 'service_date'], axis=1)  # Drop unnecessary columns
    y = (data['claim_status'] == 'denied').astype(int)  # Convert claim status to binary (1 for denied, 0 for approved)

    # Return preprocessor and processed data
    return preprocessor, X, y

def train_model(preprocessor, X, y):
    # Create a pipeline that first preprocesses the data and then trains the model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    return model

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'dataset.csv')
    model_path = os.path.join(os.path.dirname(current_dir), 'models', 'model.pkl')

    data = load_data(data_path)
    preprocessor, X, y = preprocess_training_data(data)
    model = train_model(preprocessor, X, y)
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()