from gradio import Interface, inputs, outputs
import pandas as pd
import joblib  # Import joblib directly

# Load the trained model
model = joblib.load('../models/model.pkl')

def predict_claim_status(claim_amount, cpt_code):
    # Prepare the input data for prediction
    input_data = pd.DataFrame([[claim_amount, cpt_code]], columns=['claim_amount', 'cpt_code'])
    prediction = model.predict(input_data)
    return "Denied" if prediction[0] == 1 else "Approved"

# Define the Gradio interface
iface = Interface(
    fn=predict_claim_status,
    inputs=[
        inputs.Number(label="Claim Amount"),
        inputs.Textbox(label="CPT Code")
    ],
    outputs=outputs.Textbox(label="Claim Status"),
    title="Claims Optimization and Error Detection",
    description="Enter claim details to predict if the claim will be denied or approved."
)

if __name__ == "__main__":
    iface.launch()