import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- تحميل الملفات ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("onehot_encoder.pkl", "rb") as f:
    onehot_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- الاتجاهات ---
dir_map = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

# --- الأماكن ومتوسط الأمطار ---
avg_rain_per_location = {
    'Sydney': 118, 'Melbourne': 102, 'Brisbane': 115,
    'Adelaide': 89, 'Perth': 87, 'Hobart': 110,
    'Darwin': 132, 'Canberra': 93
}

# --- تصميم الواجهة ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #cce6ff, #e6f2ff);
        color: black;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        text-align: center;
        color: black;
        margin-bottom: 10px;
    }
    label, .stSelectbox, .stNumberInput, .stButton, .stMarkdown {
        color: black !important;
    }
    .sun {
        position: absolute;
        top: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        background: radial-gradient(circle, #ffeb3b 0%, #fdd835 60%, #fbc02d 100%);
        border-radius: 50%;
        box-shadow: 0 0 20px #fdd835;
        z-index: 1;
    }
    .cloud {
        width: 150px;
        height: 80px;
        background: #fff;
        border-radius: 50%;
        position: relative;
        margin: 0 auto 20px;
        box-shadow: 60px 0 #fff, 30px -20px #fff;
    }
    </style>
    <div class="sun"></div>
    <div class="cloud"></div>
""", unsafe_allow_html=True)

# --- عنوان ---
st.title("🌦️ RainTomorrow Prediction App")

# --- إدخال المستخدم ---
Location_name = st.selectbox("🌍 Location", list(avg_rain_per_location.keys()))
Location = avg_rain_per_location[Location_name]

Rainfall = st.number_input("🌧️ Rainfall (mm)")
WindGustSpeed = st.number_input("💨 Wind Gust Speed")
WindGustDir = st.selectbox("🧭 Wind Gust Direction", list(dir_map.keys()))
WindDir9am = st.selectbox("🧭 Wind Direction at 9am", list(dir_map.keys()))
WindDir3pm = st.selectbox("🧭 Wind Direction at 3pm", list(dir_map.keys()))
Humidity9am = st.number_input("💧 Humidity at 9am")
Humidity3pm = st.number_input("💧 Humidity at 3pm")
Pressure3pm = st.number_input("🌬️ Pressure at 3pm")
Temp3pm = st.number_input("🌡️ Temperature at 3pm")
RainToday = st.selectbox("☔ Did it rain today?", ["No", "Yes"])
RISK_MM = st.number_input("📏 RISK_MM")
Month = st.selectbox("📆 Month", list(range(1, 13)))
WindSpeed_mean = st.number_input("💨 Avg Wind Speed")
Season = st.selectbox("🍂 Season", ["Autumn", "Spring", "Summer", "Winter"])

# --- تجهيز البيانات ---
RainToday_encoded = label_encoder.transform([RainToday])[0]
season_encoded = onehot_encoder.transform([[Season]])[0]

raw_input = np.array([[Location, Rainfall, dir_map[WindGustDir], WindGustSpeed,
                      dir_map[WindDir9am], dir_map[WindDir3pm],
                      Humidity9am, Humidity3pm, Pressure3pm,
                      Temp3pm, RainToday_encoded, RISK_MM,
                      Month, WindSpeed_mean]])

input_data = np.concatenate((raw_input, season_encoded.reshape(1, -1)), axis=1)
input_scaled = scaler.transform(input_data)

# --- التوقع ---
if st.button("🔍 Predict Rain Tomorrow"):
    pred = model.predict(input_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred])[0]

    if pred_label == "Yes":
        st.success("🌧️ Yes, it will rain tomorrow!")
    else:
        st.info("☀️ No, it will not rain tomorrow.")
