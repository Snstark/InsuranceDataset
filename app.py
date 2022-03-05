import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.title("ABC Inc.")
st.write("Insurance Charges Prediction ")
st.image("data//insurance.jpg")

data = pd.read_csv('data//insurance.csv')
DT = pickle.load(open('DTR.pkl', 'rb'))
RFR = pickle.load(open('RFR.pkl', 'rb'))
Lasso = pickle.load(open('LaR.pkl', 'rb'))
Ridge = pickle.load(open('Ri.pkl', 'rb'))
Linear = pickle.load(open('ILR.pkl', 'rb'))

nav = st.sidebar.radio('Navigation', ['Home', 'Prediction'])
if nav == 'Home':
    st.subheader('Insurance Charges Prediction')
    if st.checkbox('Show Data'):
        st.dataframe(data)

if nav == 'Prediction':
    st.subheader('Please give the following information:')

    age = st.number_input("Please enter the age", min_value=0.0)

    sex = st.number_input(" Please enter the gender  for Male  = 0, for  Female = 1")

    bmi = st.number_input("Please enter the BMI", min_value=0.0)

    children = st.number_input("enter the number of children the respective person has ", min_value=0.0)

    smoker = st.number_input("if the person is smoker then give 1 for YES  and 0 for NO", min_value=0.0, max_value=1.0)

    NE = st.number_input("if the person is from NorthEast then YES = 1, NO = 0")

    NW = st.number_input("if the person is from NorthWest then YES = 1, NO = 0")

    SE = st.number_input("if the person is from SouthEast then YES = 1, NO = 0")

    SW = st.number_input("if the person is from SouthWest then YES = 1, NO = 0")

    x = np.array([age, sex, bmi, children, smoker, NE, NW, SE, SW])

    x = x.reshape(1, 9)

    st.header("Select the REGRESSOR Algorithms")
    Cls = st.sidebar.radio('Algorithms', ['DecisionTreeRegressor', 'RandomForestRegressor',
                                          "Linear Regression", "Lasso Regression", "Ridge Regression", "ALL"])
    if Cls == "DecisionTreeRegressor":
        if st.button('Predict'):
            st.write("DECISION TREE REGRESSOR")
            S = st.title('The predicted Insurance Charges are ' + str(round(int(DT.predict(x)[0]))) + '$')

    if Cls == "RandomForestRegressor":
        if st.button('Predict'):
            st.write("RANDOM FOREST REGRESSOR")
            S = st.title(f'The predicted insurance Charges are  ' + str(round(int(RFR.predict(x)[0]))) + '$')

    if Cls == "Ridge Regression":
        if st.button('Predict'):
            st.write("Ridge Regression")
            S = st.title(f'The predicted insurance Charges are  ' + str(round(int(Ridge.predict(x)[0]))) + '$')

    if Cls == "Lasso Regression":
        if st.button('Predict'):
            st.write("Lasso Regression")
            S = st.title(f'The predicted insurance Charges are  ' + str(round(int(Lasso.predict(x)[0]))) + '$')

    if Cls == "Linear Regression":
        if st.button('Predict'):
            st.write("Linear")
            S = st.title(f'The predicted insurance Charges are  ' + str(round(int(Linear.predict(x)[0]))) + '$')

    if Cls == "ALL":
        if st.button('Predict'):
            st.title(
                f'The predicted insurance charges using Decision Tree Regressor ' + str(
                    round(int(DT.predict(x)[0]))) + '$')
            st.title(
                f'The predicted Insurance Charges using RANDOM FOREST REGRESSOR  is' + str(
                    round(int(RFR.predict(x)[0]))) + '$')
            st.title(f'the predicted  insurance charges using Linear regressor is' + str(
                round(int(Linear.predict(x)[0]))) + '$')
            st.title(f'the predicted  insurance charges using Ridge regressor is' + str(
                round(int(Ridge.predict(x)[0]))) + '$')
            st.title(f'the predicted  insurance charges using Lasso regressor is' + str(
                round(int(Lasso.predict(x)[0]))) + '$')
