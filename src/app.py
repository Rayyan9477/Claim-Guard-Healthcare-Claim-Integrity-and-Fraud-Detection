from model_training import load_model, preprocess_prediction_data
import gradio as gr
import pandas as pd
import os
import numpy as np

def predict_claim_status(claim_amount, cpt_code):
    # Create input data
    input_data = pd.DataFrame({
        'claim_amount': [float(claim_amount)],
        'cpt_code': [str(cpt_code)]
    })
    
    # Load model and feature names
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(current_dir), 'models', 'model.pkl')
    model, feature_names = load_model(model_path)
    
    # Preprocess input
    X = preprocess_prediction_data(input_data)
    
    # Align features with training data
    missing_cols = set(feature_names) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[feature_names]
    
    # Make prediction
    prediction = model.predict(X)
    return "Denied" if prediction[0] == 1 else "Approved"

def main():
    interface = gr.Interface(
        fn=predict_claim_status,
        inputs=[
            gr.Number(label="Claim Amount"),
            gr.Textbox(label="CPT Code")
        ],
        outputs=gr.Textbox(label="Prediction"),
        title="Claims Optimization System",
        description="Enter claim details to predict approval status"
    )
    interface.launch(share=True)

if __name__ == "__main__":
    main()