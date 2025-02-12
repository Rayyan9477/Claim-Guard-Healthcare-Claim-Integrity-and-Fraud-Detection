# ClaimGuard: Healthcare Claim Integrity and Fraud Detection

## Description

ClaimGuard is a smart system designed to help medical billing companies and hospitals ensure the integrity of their healthcare claims and prevent denials. Think of it as a sophisticated tool that analyzes claim data to predict whether an insurance company is likely to reject a claim *before* it's even submitted.

**What does ClaimGuard do?**

*   **Predicts Claim Denials:** Uses predictive analytics to identify potential issues that could lead to claim denials.
*   **Highlights Errors:** Points out specific areas of concern, such as incorrect coding, missing information, or the need for prior authorization.
*   **Provides Recommendations:** Suggests actions to take to fix the errors and increase the chances of approval.

**How does ClaimGuard benefit medical billing companies and hospitals?**

*   **Reduces Claim Denials:** By catching errors early, ClaimGuard helps prevent costly claim denials and rework.
*   **Accelerates Payments:** Faster claim approvals mean quicker payments and improved cash flow.
*   **Increases Revenue:** By minimizing denials and maximizing approvals, ClaimGuard helps boost revenue for healthcare providers.
*   **Improves Efficiency:** Automates the error detection process, freeing up staff to focus on other important tasks.

## Project Structure

*   **data/**: Contains the dataset used for analysis and model training.
    *   `dataset.csv`: Historical claims data including features such as patient age, gender, insurance provider, service codes, diagnosis codes, claim amounts, and claim statuses.
*   **models/**: Stores the trained predictive model.
    *   `model.pkl`: The trained machine learning model (Random Forest Classifier) that predicts claim denial risk.
*   **notebooks/**: Contains Jupyter notebooks for data analysis.
    *   `data_analysis.ipynb`: Exploratory data analysis (EDA) and visualizations to identify patterns in claim denials and errors.
*   **src/**: Source code for the application.
    *   `app.py`: Main application script, initializes the Streamlit user interface.
    *   `model_training.py`: Functions for training the predictive model and saving it.
*   **requirements.txt**: Lists the Python libraries required to run the project.

## Tech Stack

*   **Streamlit:** For creating the user-friendly web interface.
*   **pandas:** For data manipulation and analysis.
*   **scikit-learn:** For machine learning model training and prediction.
*   **matplotlib and seaborn:** For data visualization.
*   **joblib:** For saving and loading the trained model.

## Setup Instructions

1.  Clone the repository:

    ```bash
    git clone <repository-url>
    cd claims-optimization
    ```

2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3.  Run the application:

    ```bash
    streamlit run src/app.py
    ```

## Usage Guidelines

1.  **Access the ClaimGuard App:** Once the app is running, it will open in your web browser.
2.  **Input Claim Details:** Enter the claim details in the input fields provided.
3.  **Predict Claim Status:** Click the "Predict Claim Status" button to get a prediction.
4.  **Review the Prediction:** The app will display whether the claim is likely to be "Approved" or "Denied".
5.  **Explore Data Analysis:** Use the sidebar to toggle the Exploratory Data Analysis (EDA) section. This section provides visualizations to help you understand the data.

## Contributing

We welcome contributions to make ClaimGuard even better! If you have ideas for new features, improvements, or bug fixes, please fork the repository and submit a pull request.

## Contact

**Rayyan Ahmed**

*   GitHub: [https://github.com/Rayyan9477](https://github.com/Rayyan9477)
*   LinkedIn: [https://www.linkedin.com/in/rayyan-ahmed9477/](https://www.linkedin.com/in/rayyan-ahmed9477/)
*   Email: rayyanahmed265@yahoo.com