import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# LOAD DATA (IMPORTANT FIX)
# -----------------------------
data = pd.read_csv("Crop_recommendation.csv")

# Fix: split single column if needed
if len(data.columns) == 1:
    data = data.iloc[:, 0].str.split(",", expand=True)

# Assign proper column names
data.columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]

# Convert numeric columns
for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
    data[col] = pd.to_numeric(data[col])

# -----------------------------
# SPLIT DATA
# -----------------------------
X = data.drop("label", axis=1)
y = data["label"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🌱 AI Smart Agriculture System")
st.write("Enter soil and weather conditions:")

N = st.number_input("Nitrogen (N)")
P = st.number_input("Phosphorus (P)")
K = st.number_input("Potassium (K)")
temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH Value")
rainfall = st.number_input("Rainfall")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Crop"):
    prediction = model.predict([[N, P, K, temp, humidity, ph, rainfall]])
    crop = prediction[0]

    # Profit estimation (simple logic)
    profit_map = {
        "rice": 20000,
        "wheat": 18000,
        "maize": 22000,
        "cotton": 30000,
        "sugarcane": 35000
    }

    profit = profit_map.get(crop, "Not available")

    st.success(f"🌾 Recommended Crop: {crop}")
    st.info(f"💰 Estimated Profit: {profit}")