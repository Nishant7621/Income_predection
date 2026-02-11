import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# ===================== LOAD MODEL & COLUMNS =====================
model = pickle.load(open("model.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))

# ===================== UI =====================
st.header("𝑰𝒏𝒄𝒐𝒎𝒆 𝑷𝒓𝒆𝒅𝒊𝒄𝒕𝒊𝒐𝒏 𝑾𝒆𝒃 𝑨𝒑𝒑")

image = Image.open("income.jpg")
st.image(image, use_column_width=True)

# ===================== USER INPUT =====================
def user_input_features():
    age = st.slider("Age", 17, 90)

    workclass = st.selectbox(
        "Work Class",
        (
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
            "Local-gov", "State-gov", "Without-pay", "Never-worked"
        )
    )

    education_num = st.slider("Education Num", 1, 16)

    marital_status = st.selectbox(
        "Marital Status",
        (
            "Married-civ-spouse", "Divorced", "Never-married", "Separated",
            "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        )
    )

    occupation = st.selectbox(
        "Occupation",
        (
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
            "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
            "Transport-moving", "Priv-house-serv", "Protective-serv",
            "Armed-Forces"
        )
    )

    relationship = st.selectbox(
        "Relationship",
        (
            "Not-in-family", "Husband", "Wife", "Own-child",
            "Unmarried", "Other-relative"
        )
    )

    race = st.selectbox(
        "Race",
        (
            "White", "Black", "Asian-Pac-Islander",
            "Amer-Indian-Eskimo", "Other"
        )
    )

    sex = st.selectbox("Sex", ("Male", "Female"))

    capital_gain = st.slider("Capital Gain", 0, 100000)
    capital_loss = st.slider("Capital Loss", 0, 5000)
    hours_per_week = st.slider("Hours per Week", 1, 90)

    country = st.selectbox(
        "Country",
        (
            "United-States", "India", "Mexico", "Philippines",
            "Germany", "Canada", "England", "China", "Japan",
            "South", "Puerto-Rico", "Cuba"
        )
    )

    education_level = st.selectbox(
        "Education Level",
        (
            "Compulsory", "High_School_grad", "Associate",
            "Bachelors", "Masters", "Professor"
        )
    )

    data = {
        "age": age,
        "workclass": workclass,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "country": country,
        "education_level": education_level,
    }

    return pd.DataFrame(data, index=[0])

# ===================== PREDICTION =====================
def predict_income(input_df):
    # One-hot encode
    df = pd.get_dummies(input_df, drop_first=True)

    # Add missing columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure same column order
    df = df[model_columns]

    return model.predict(df)[0]

# ===================== RUN APP =====================
input_df = user_input_features()
st.subheader("User Input")
st.write(input_df)

if st.button("Predict"):
    result = predict_income(input_df)

    if result == 0:
        st.success("💰 Person earns **LESS than 50K**")
    else:
        st.success("💸 Person earns **MORE than 50K**")
