import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import numpy as np
#pipe=pickle.load(open('automl.pkl','rb'))
#df=pickle.load(open(r"D:\New folder\New folder\df.pkl",'rb'))
#pipe=pickle.load(open(r"D:\New folder\New folder\best_decision_tree_model.pkl",'rb'))


df=pickle.load(open("df.pkl",'rb'))
pipe=pickle.load(open("best_decision_tree_model.pkl",'rb'))
    
st.title("WC2023 Win Predictor")
Team_A=st.selectbox('Team_A', sorted(df.Team_A.unique()))
Team_B=st.selectbox('Team_B', sorted(df.Team_B.unique()))
Toss=st.selectbox('Toss', sorted(df.Toss.unique()))
Home_or_away=st.selectbox('Home_or_away', sorted(df['Home/Away'].unique()))
Chasing_or_defending=st.selectbox('Chasing_or_defending', sorted(df['Chasing / Defending'].unique()))
Day_or_Night=st.selectbox('Day_or_Night', sorted(df['Day/Night'].unique()))
Venue=st.selectbox('Venue', sorted(df.Venue.unique()))



input_reshped=pd.DataFrame({'Team_A':[Team_A],'Team_B':[Team_B],'Toss':[Toss],'Home/Away':[Home_or_away],'Chasing / Defending':[Chasing_or_defending], 'Day/Night':[Day_or_Night], 'Venue':[Venue]})
predict=pipe.predict(input_reshped)
print('prediction')
print(predict)


if st.button('Predict'):
    st.balloons()
    if predict[0] == 0:
        result = f"{Team_A}  will win"
    else:
        result = f"{Team_B}  will win"
    st.success(result)