# importing dependencies
from flask import Flask,request,jsonify,render_template
import numpy as np
import shap
import pickle
import bz2
import configparser
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser





# Load the compressed model
with bz2.BZ2File("modelbz2.pkl.bz2", "rb") as f:
    model = pickle.load(f)


# input preparation
import numpy as np

def process_form_input(values):
    """
    Converts input list into a numpy array of length 28.
    
    Parameters:
    values (list): List of input values from the form, ordered as:
    ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
     'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'HasMortgage',
     'HasDependents', 'HasCoSigner', 'Default', 'EmploymentType',
     'MaritalStatus', 'LoanPurpose', 'Education']
    
    Returns:
    np.ndarray: Processed numpy array of shape (28,)
    """
    
    # Initialize an array of zeros with length 28
    features = np.zeros(28)
    
    # Assign numerical values directly
    features[:9] = values[:9]  # ['Age', 'Income', 'LoanAmount', ..., 'DTIRatio']
    
    # Binary categorical features
    features[9] = 1 if values[9] == 'Yes' else 0  # HasMortgage
    features[10] = 1 if values[10] == 'Yes' else 0  # HasDependents
    features[11] = 1 if values[11] == 'Yes' else 0  # HasCoSigner
    
    # Employment Type One-Hot Encoding
    employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
    features[12:16] = [1 if values[12] == emp else 0 for emp in employment_types]
    
    # Marital Status One-Hot Encoding
    marital_statuses = ['Divorced', 'Married', 'Single']
    features[16:19] = [1 if values[13] == status else 0 for status in marital_statuses]
    
    # Loan Purpose One-Hot Encoding
    loan_purposes = ['Auto', 'Business', 'Education', 'Home', 'Other']
    features[19:24] = [1 if values[14] == purpose else 0 for purpose in loan_purposes]
    
    # Education One-Hot Encoding
    education_levels = ["Bachelor's", 'High School', "Master's", 'PhD']
    features[24:28] = [1 if values[15] == level else 0 for level in education_levels]
    
    return features


# fetching feature importance using 
def get_features_imp_using_shap(model,inp):
    explainer = shap.Explainer(model)
    shap_values = explainer(inp)
    return np.abs(shap_values.values).mean(axis=1)
    
    

def LLM_connection(feature_importance_dict,model_prediction,user_input):
    config = configparser.ConfigParser()
    config.read('config.ini')
    groq = config['groq']
    os.environ['GROQ_API_KEY'] = groq.get('GROQ_API_KEY')

    messages = [
    SystemMessage(content='''You are a Senior Underwriter at an NBFC responsible for assessing loan applications.
    You have conducted a thorough analysis considering 27 key factors that influence default risk. 
    You have been provided with the feature importance scores from a predictive model. 

    The model's prediction is as follows:
    - 1 indicates the applicant is predicted to default.
    - 0 indicates the applicant is predicted not to default.

    ### Formatting Instructions:
    This content will be displayed inside an HTML page within a container with limited width (600px) and a scrollable text box.  
    Ensure the report is **concise, well-structured, and easy to read** while keeping text within readable limits.

    **Use the following structure:**
    1. **Introduction** : A brief summary of the applicant's profile.
    2. **Key Risk Indicators** : List the top 3-5 contributing factors with a short explanation.
    3. **Analysis & Justification** : Provide a clear and **concise** explanation of the model's decision in 4-6 sentences.
    4. **Recommendation** : Clearly state whether the loan should be **approved** or **rejected**, with a strong supporting rationale.

    **Formatting Guidelines:**
    - Use **short paragraphs** (max 2-3 lines per paragraph) to ensure readability.
    - Use **bullet points** for key indicators.
    - **Avoid overly long text blocks**, as the report is displayed inside a limited-height box.
    - Ensure professional, formal tone.

    Output an **easy-to-read, well-formatted report** that fits neatly within the provided HTML structure.'''),


        HumanMessage(content=f'''Provide a formal report evaluating a loan application based on the given model's predictions. 

    ### Loan Assessment Report:
    - **Applicant Overview:** A brief summary of the applicant's profile.
    - **Key Risk Indicators:** The top contributing factors influencing the model's decision.
    - **Analysis & Justification:** A reasoned explanation of why the model predicted a default (or non-default).
    - **Recommendation:** A final underwriting decision with supporting rationale.

    ### Additional Information:
    - **User inputs:** {user_input} and interpret them as values of ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
     'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'HasMortgage',
     'HasDependents', 'HasCoSigner', 'Default', 'EmploymentType',
     'MaritalStatus', 'LoanPurpose', 'Education']
    - **Feature Importance Scores:** {feature_importance_dict}
    - **Model Prediction:** model_prediction
    ''')
    ]
    parser = StrOutputParser()
    model = ChatGroq(model="llama3-8b-8192")

    chain = model | parser
    return chain.invoke(messages)




app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    form_features=[x for x in request.form.values()]
    model_input_features=process_form_input(form_features)

    columnNames=['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
       'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'HasMortgage',
       'HasDependents', 'HasCoSigner', 'EmploymentType_Full-time',
       'EmploymentType_Part-time', 'EmploymentType_Self-employed',
       'EmploymentType_Unemployed', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'MaritalStatus_Single', 'LoanPurpose_Auto',
       'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home',
       'LoanPurpose_Other', "Education_Bachelor's", 'Education_High School',
       "Education_Master's", 'Education_PhD']
    
    prediction = model.predict_proba(model_input_features.reshape(1, -1))
    output = 'Defaulter' if prediction[0][1] >=prediction[0][1]  else 'Not a Defaulter'
    
    feature_importance=get_features_imp_using_shap(model,model_input_features)

    # features after one hot encoding
    feature_importance_dict = dict(zip(columnNames, feature_importance))


    # LLM summarization
    report =LLM_connection(feature_importance_dict,output,form_features)

    return render_template('result.html', prediction=output,report=report,predict_prob=f"{prediction[0][1]:.2%}")

if __name__ == "__main__":
    app.run()