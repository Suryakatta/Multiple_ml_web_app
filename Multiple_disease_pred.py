# -*- coding: utf-8 -*-
"""
Multiple Disease Prediction System (Fixed Version)
"""

import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu 

# ================= LOAD MODELS =================
diabetes_model = pickle.load(open('C:/Users/steja/OneDrive/Desktop/Multiple disease prediction/saved_model/train_models.sav','rb'))
HeartD_model = pickle.load(open('C:/Users/steja/OneDrive/Desktop/Multiple disease prediction/saved_model/heart_train.sav','rb'))
parkin_model = pickle.load(open('C:/Users/steja/OneDrive/Desktop/Multiple disease prediction/saved_model/parkinson_disease.sav','rb'))
parkin_scaler = pickle.load(open('C:/Users/steja/OneDrive/Desktop/Multiple disease prediction/saved_model/parkinsons_scaler.sav','rb'))

# ================= SIDEBAR =================
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction','Heart Disease Prediction','Parkinsons Prediction'],
        icons=['activity','heart','person'],
        default_index=0
    )

# ================= DIABETES =================
if selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction')

    Pregnancies = st.number_input('Number of Pregnancies')
    Glucose = st.number_input('Glucose Level')
    BloodPressure = st.number_input('Blood Pressure')
    SkinThickness = st.number_input('Skin Thickness')
    Insulin = st.number_input('Insulin Level')
    BMI = st.number_input('BMI')
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function')
    Age = st.number_input('Age')

    if st.button('Diabetes Test Result'):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])

        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            st.error('⚠️ The person is Diabetic')
        else:
            st.success('✅ The person is NOT Diabetic')

# ================= HEART =================
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction')

    age = st.number_input('Age')
    sex = st.selectbox('Sex', [0,1])  # 0 = Female, 1 = Male
    cp = st.number_input('Chest Pain Type (0-3)')
    trestbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Cholesterol')
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0,1])
    restecg = st.number_input('Resting ECG (0-2)')
    thalach = st.number_input('Max Heart Rate Achieved')
    exang = st.selectbox('Exercise Induced Angina', [0,1])
    oldpeak = st.number_input('ST depression')
    slope = st.number_input('Slope (0-2)')
    ca = st.number_input('Number of Major Vessels (0-4)')
    thal = st.number_input('Thal (0-3)')

    if st.button('Heart Test Result'):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]])

        prediction = HeartD_model.predict(input_data)

        if prediction[0] == 1:
            st.error('⚠️ The person has Heart Disease')
        else:
            st.success('✅ The person is Healthy')

# ================= PARKINSONS =================
if selected == 'Parkinsons Prediction':
    st.title('🧠 Parkinsons Prediction')

    fo = st.number_input('MDVP:Fo(Hz)')
    fhi = st.number_input('MDVP:Fhi(Hz)')
    flo = st.number_input('MDVP:Flo(Hz)')
    jitter_percent = st.number_input('MDVP:Jitter(%)')
    jitter_abs = st.number_input('MDVP:Jitter(Abs)')
    rap = st.number_input('MDVP:RAP')
    ppq = st.number_input('MDVP:PPQ')
    ddp = st.number_input('Jitter:DDP')
    shimmer = st.number_input('MDVP:Shimmer')
    shimmer_db = st.number_input('MDVP:Shimmer(dB)')
    apq3 = st.number_input('Shimmer:APQ3')
    apq5 = st.number_input('Shimmer:APQ5')
    apq = st.number_input('MDVP:APQ')
    dda = st.number_input('Shimmer:DDA')
    nhr = st.number_input('NHR')
    hnr = st.number_input('HNR')
    rpde = st.number_input('RPDE')
    dfa = st.number_input('DFA')
    spread1 = st.number_input('spread1')
    spread2 = st.number_input('spread2')
    d2 = st.number_input('D2')
    ppe = st.number_input('PPE')

    if st.button('Parkinson Test Result'):
        input_data = np.array([[fo,fhi,flo,jitter_percent,jitter_abs,rap,ppq,ddp,
                                shimmer,shimmer_db,apq3,apq5,apq,dda,nhr,hnr,
                                rpde,dfa,spread1,spread2,d2,ppe]])

        # Apply scaler
        input_data = parkin_scaler.transform(input_data)

        prediction = parkin_model.predict(input_data)

        if prediction[0] == 1:
            st.error('⚠️ The person has Parkinson’s Disease')
        else:
            st.success('✅ The person is Healthy')