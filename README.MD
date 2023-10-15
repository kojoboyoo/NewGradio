# Customer Attrition Prediction App
Customer Attrition Prediction App is a web application built with Gradio that predicts customer churn based on input data. Churn prediction is a crucial task for businesses as it helps in understanding customer behavior and aids in customer retention strategies


 

## Features
* User-Friendly Interface: The app provides an intuitive and easy-to-use interface for users to input customer data and receive churn predictions.

 

* Accurate Predictions: Utilizes a pre-trained Random Forest Classifier model to ensure accurate churn predictions based on the provided customer information.

 

* Interactive Visualization: Displays predictions in a clear and understandable format, making it easier for users to interpret the results.

 

## Installation
* clone [repository](https://github.com/kojoboyoo/Newgradio.git)
* cd into file path where files are
* Install Dependencies:pip install -r requirements.txt
* Run the App: python customer_churn_fin.py. The app will be accessible at [local host](http://127.0.0.1:7860)

 

## Input Features
The following input features are required for making predictions:
* Total Charges ($)
* Monthly Charges ($)
* Number of months being on the network
* Gender (Male/Female)
* Senior or non-Senior citizen (Yes/No)
* Partner (Yes/No)
* Dependents (Yes/No)
* Phone services (Yes/No)
* Multiple Lines (Yes/No)
* Internet Service (DSL/Fiber optic/No)
* Online Security (Yes/No/No internet service)
* Online Backup (Yes/No/No internet service)
* Device Protection (Yes/No/No internet service)
* Tech Support (Yes/No/No internet service)
* Streaming TV (Yes/No/No internet service)
* Streaming Movies (Yes/No/No internet service)
* Contract Term (Month-to-month/One year/Two years)
* Paperless Billing (Yes/No)
* Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))

 

### Input Customer Data:
Enter customer details such as total charges, monthly charges, tenure, gender, and other relevant information into the provided fields.

 

### Get Churn Prediction:
As you input the details, the app will display whether the customer is likely to churn or not based on the input data.

 

## Pipeline and Model
* A pipeline was created to encode the numerical and categorical variables in the dataset for easy modeling.
* The app uses a Random Forest Classifier trained on the given dataset to make predictions.


 

## Acknowledgements
We would like to thank Azubi Africa for the opportunity to learn how to build an app with Gradio and the team who made this possible inspite of the setbacks.

 

## Authors
| Name | GitHub link |
| ---- | ---- |
| Enoch Taylor-Nketiah         | https://github.com/kojoboyoo       |

 

 

| Project |	Name |	Published Article |	
| ---- | -----| ----- | 
| LP4	| Gradio App |	[Gradio App LP4](https://www.linkedin.com/pulse/building-interactive-machine-learning-interfaces-enoch-taylor-nketiah%3FtrackingId=WnGdnHocTci91Z2u6K9VjQ%253D%253D/?trackingId=WnGdnHocTci91Z2u6K9VjQ%3D%3D) | 