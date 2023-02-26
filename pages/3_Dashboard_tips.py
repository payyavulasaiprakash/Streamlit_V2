import pandas as pd
import streamlit as st
import os
import ntpath
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

st.title("Dashboard - Tips dataset analysis and tip prediction using linear regression")


file_dir,_=ntpath.split((os.path.abspath(__file__)))
dir=os.path.join(os.path.dirname(file_dir),'resources')
dataset_file_path = os.path.join(dir, "data", "tips.csv")

st.subheader("Sample of dataset")

dataset=pd.read_csv(dataset_file_path)

dataset['smoker']=dataset['smoker'].map({"No":0,"Yes":1})
dataset['sex']=dataset['sex'].map({"Female":0,"Male":1})

st.dataframe(dataset.head())

print(dataset.columns)

X=dataset[['total_bill','smoker','sex']]
y=dataset['tip']

users_input=['total_bill','smoker','sex']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
accuracy=r2_score(predictions,y_test)

bill= st.text_input("Enter the bill in number: ")
st.write('The bill amount is', bill)
smoker = st.selectbox("Smoker or not: ",dataset.smoker.unique())
sex = st.selectbox("Sex : ",dataset.smoker.unique())

predicted_value=model.predict(np.array([bill,smoker,sex]).reshape(-1,3))
st.write("Predict Tip is of ",predicted_value)

st.write("Thank you for using the machine learning for predicting the tip")
