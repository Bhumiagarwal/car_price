import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

# Load data (make sure CSV is in same folder)
car = pd.read_csv("cleaned_car.csv")

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Predictor")
st.write("Predict the price of a car using trained ML model")

# Input fields
company = st.selectbox(
    "Select Company",
    sorted(car["company"].unique())
)

car_model = st.selectbox(
    "Select Car Model",
    sorted(car["name"].unique())
)

year = st.selectbox(
    "Select Year",
    sorted(car["year"].unique(), reverse=True)
)

fuel_type = st.selectbox(
    "Select Fuel Type",
    car["fuel_type"].unique()
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    step=1000
)

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame(
        [[car_model, company, year, kms_driven, fuel_type]],
        columns=["name", "company", "year", "kms_driven", "fuel_type"]
    )

    prediction = model.predict(input_df)

    st.success(f"Estimated Car Price: ₹ {np.round(prediction[0], 2)}")
