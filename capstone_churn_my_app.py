# Streamlit Documentation: https://docs.streamlit.io/


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image  # to deal with images (PIL: Python imaging library)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder



st.header('This is a employee churn analysis prediction app.')

st.text("Select your employee features on side bar and app will return your car price.")



st.sidebar.header("Chose your car features...")



# To load machine learning model
import pickle
filename = "RF_Grid_Capstone2DS_Model.pkl"
model = pickle.load(open(filename, "rb"))

df = pd.read_csv('HR_Dataset.csv')

departments_u = df['departments'].drop_duplicates()
salary_u = df['salary'].drop_duplicates()


# To take feature inputs
departments = st.sidebar.selectbox("Select your employee departments...:", departments_u)
salary = st.sidebar.selectbox("Select your employee salary type...:", salary_u)
satisfaction_level = st.sidebar.slider("Select your employee satisfaction level 0 to 1...:",min_value=0, max_value=1, step=0.01)
last_evaluation = st.sidebar.slider("Select your employee last evaluation 0 to 1...:",min_value=0.36, max_value=1, step=0.01)
number_project = st.sidebar.slider("Select how many of projects assigned to an employee ...:",min_value=2, max_value=7, step=1)
average_montly_hours = st.sidebar.number_input("Select your car mileage...:",min_value=90, max_value=310)
time_spend_company = st.sidebar.slider("Select how many of projects assigned to an employee ...:",min_value=2, max_value=10, step=1)
work_accident = st.sidebar.slider("Select employee any work accident...:",min_value=0, max_value=1, step=1)
promotion_last_5years = st.sidebar.slider("Select employee any promotion last 5 years...:",min_value=0, max_value=1, step=1)


# Create a dataframe using feature inputs
new_sample = {
    'departments'             : departments,
    'salary'                  : salary,
    'satisfaction_level'      : satisfaction_level,
    'last_evaluation'         : last_evaluation,
    'number_project'          : number_project,
    'average_montly_hours'    : average_montly_hours,
    'time_spend_company'      : time_spend_company,
    'time_spend_company'      : time_spend_company,
    'work_accident'           : work_accident
}

df = pd.DataFrame.from_dict([new_sample])
st.table(df)

# Prediction with user inputs
predict = st.button("Predict")
result = model.predict(df)
if predict :
    st.success(result[0])
    st.balloons()


