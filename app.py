import gradio as gr
import pickle
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# copy all necessary features for encoding and scaling in the same sequence from your python codes
expected_inputs = ["TotalCharges","MonthlyCharges","Tenure","Gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
#group the numeric and categorical features exactly as was done in the python codes
numerical = ["TotalCharges","MonthlyCharges","Tenure"]
categoricals = ["Gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]




# Load your pipeline, Scaler, and encoder
def load_pipeline(file_path):
    with open(file_path, "rb") as file:
        pipeline = pickle.load(file)
    return pipeline

# Load the pipeline (assuming it contains preprocessing steps)
pipeline = load_pipeline(r"C:\Users\etnketiah\Documents\New_gradio\pipeline1.pkl")

# Load your RandomForestClassifier model separately
with open(r'C:\Users\etnketiah\Documents\New_gradio\random_forest_classifier.pkl', "rb") as file:
    randf_classifier = pickle.load(file)


def predict_churn(*args, model=randf_classifier, pipeline=pipeline):
    try:
        # Create a DataFrame with input data
        input_data = pd.DataFrame([args], columns=expected_inputs)

        # Preprocess the input data using the loaded pipeline
        preprocessed_data = pipeline.transform(input_data)

        # Make predictions using the RandomForestClassifier model
        model_output = model.predict_proba(preprocessed_data)[:, 1]  # Use predict_proba to get probabilities

        # Convert model output to a single probability value
        probability_of_churn = model_output[0]

        # You can adjust the threshold as needed
        threshold = 0.5

        # Compare the probability with the threshold to make a binary prediction
        if probability_of_churn > threshold:
            return "Churn"
        else:
            return "No Churn"
    except Exception as e:
        # Handle exceptions gracefully
        return f"Error: {str(e)}"
    

    





# Define expected_inputs, numerical, and categoricals here
TotalCharges= gr.Number(label="Total Charges for Customer")
MonthlyCharges = gr.Number(label="Monyhly Charges for Customer")
Tenure= gr.Number(label="How long has the Customer been on the network in months",minimum=1)
Gender = gr.Radio(label="What is the Gender of the Customer", choices=["Male", "Female"])
SeniorCitizen = gr.Radio(label="Is the Customer a Senior or non Senior citizen?", choices=["Yes", "No"])
Partner = gr.Radio(label="Is the Customer having a Partner?", choices=["Yes", "No"])
Dependents = gr.Radio(label="Is the Customer having a Dependants?", choices=["Yes", "No"])
PhoneService = gr.Radio(label="Is the Customer any Phone services?", choices=["Yes", "No"])
MultipleLines = gr.Radio(label="Is the Customer having Multiple Lines?", choices=["Yes", "No"])
InternetService = gr.Radio(label="Is the Customer having Internet Service?", choices=["DSL", "Fiber optic", "No"])
OnlineSecurity = gr.Radio(label="Is the Customer having Multiple Lines", choices=["Yes", "No", "No internet service"])

OnlineBackup = gr.Radio(label="Has the Customer requested for Online backup?", choices=["Yes", "No", "No internet service"])
DeviceProtection = gr.Radio(label="Is the Customer having Device Protection?", choices=["Yes", "No", "No internet service"])
TechSupport = gr.Radio(label="Is the Customer having Tech Support?", choices=["Yes", "No", "No internet service"])
StreamingTV = gr.Radio(label="Is the Customer having StreamingTV?", choices=["Yes", "No", "No internet service"])
StreamingMovies = gr.Radio(label="Is the Customer having Streaming Movies option?", choices=["Yes", "No", "No internet service"])
Contract = gr.Radio(label="Which Contract is the Customer on?", choices=["Month-to-month", "One year", "Two year"])
PaperlessBilling = gr.Radio(label="Is the Customer on Paperless Billing?", choices=["Yes", "No"])
PaymentMethod = gr.Radio(label="Is the Customer having Multiple Lines", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])


# Define the Gradio interface
gr.Interface(
    inputs=[TotalCharges, MonthlyCharges, Tenure, Gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod],
    fn=predict_churn,
    outputs=gr.outputs.Label("Prediction"),
    title="Customer Attrition Prediction App",
    description="Enter customer information to predict churn.",
    live=True  # Set to True to run the app interactively
).launch(share=True,inbrowser=True)




