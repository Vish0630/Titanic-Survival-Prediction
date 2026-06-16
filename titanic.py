
import streamlit as st
import numpy as np
import joblib

st.title('🚢 TITANIC SURVIVAL PREDICTION')
st.subheader('Prediction based on Passenger Details')

# Loading Trained model and scaler
model=joblib.load('model.pkl')
scaler=joblib.load('scaler.pkl')

## User Inputs

# Creating Columns for Streamlit
col1,col2,col3=st.columns(3)

with col1:
    Pclass=st.radio('Passenger Class',[1,2,3],horizontal=True)
    Age=st.number_input('Age', min_value=0, max_value=100, step =1,value=23)

with col2:
    Sex=st.radio('Gender',['Male','Female'],horizontal=True)
    SibSp=st.number_input('Siblings/Spouse Aboard', min_value=0,max_value=10)
    Fare=st.number_input('Fare', min_value=0.0, value =32.00)

with col3:
    Embarked=st.radio('Embarked',['C','Q','S'],horizontal=True)
    Parch=st.number_input('Parent/Children Aboard', min_value=0,max_value=10)

# Encoding
sex_encoded=1 if Sex=='Female' else 0

Embarked_Q=1 if Embarked=='Q' else 0
Embarked_S=1 if Embarked=='S' else 0

# Creating Array
input_data=np.array([[Pclass, sex_encoded, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S]])

# Predict
if st.button('Predict Survival 🔮'):

    scaled_input=scaler.transform(input_data)
    prediction=model.predict(scaled_input)[0]
    probability=model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.success('Passenger Survived 🎊')
    else:
        st.error('Passenger did not Survive 😵')

    st.write(f'Probability of Survival : {probability:.2%}')
