import pandas as pd
import streamlit as st
import joblib
 
model=joblib.load('diabetes_check/svm_disbities.pkl')
scale=joblib.load('diabetes_check/scaler.pkl')
columns=joblib.load('diabetes_check/columns.pkl')
st.title("diabetes_checking")
st.markdown("provide the following details")
age=st.slider("AGE",18,100,40)
sex=st.selectbox("SEX",['M','f'])
# BP=st.number_input("Blood_Pressure",20,90,120)
BP = st.number_input("Blood_Pressure", 20, 180, 80)

# glucose=st.number_input('Glucose',20,90,200)
glucose = st.number_input('Glucose', 20, 300, 120)

# Skin_Thickness_level=st.selectbox('Skin_Thickness',0,50,100)
# This should be either slider or selectbox with a list:
Skin_Thickness_level = st.slider('Skin Thickness', 0, 100, 50)
Insulin_level = st.number_input('Insulin', 0, 900, 150)

# BMI_level=st.number_input('BMI',0,25,50)
BMI_level = st.number_input('BMI', 0.0, 70.0, 25.0)

# diabetes_Pedigree_Function=st.number_input('Diabetes_Pedigree_Function',100,300,600)
# This is for float input. Range is wrong.
diabetes_Pedigree_Function = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.5)

pregnancies=st.number_input('Pregnancies',0,10)

if st.button("Predict"):
    raw_input = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "Blood_Pressure": BP,
        "Skin_Thickness": Skin_Thickness_level,
        "Insulin": Insulin_level,
        "BMI": BMI_level,
        "Diabetes_Pedigree_Function": diabetes_Pedigree_Function,
        "Age": age
    }

    input_df = pd.DataFrame([raw_input])

    # Ensure all required columns are present
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with 0

    input_df = input_df[columns]  # Reorder columns to match training

    # Scale and predict
    scaled_input = scale.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Output result
    if prediction == 1:
        st.error("⚠️ High risk of diabetes")
    else:
        st.success("✅ Low risk of diabetes")
