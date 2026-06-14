import streamlit as st
import numpy as np
import joblib

# 🎯 Application Title
st.title("Mobile Price Prediction Application")

# 🖼️ Background image (mobile theme)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1511707171634-5f897ff02aa9"); /* mobile phones */
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.7);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# 🎨 Custom text color styling (dark text)
text_color_css = """
<style>
h1, h2, h3, h4, h5, h6, p, label, div {
    color: #000000 !important;  /* black text */
}
</style>
"""
st.markdown(text_color_css, unsafe_allow_html=True)
# 🎨 Apply light blue background to all input fields
input_style = """
<style>
div[data-testid="stNumberInput"] {
    background-color: #d9edf7;   /* light blue background */
    padding: 10px;
    border-radius: 8px;
}
div[data-testid="stSelectbox"] {
    background-color: #d9edf7;   /* light blue background */
    padding: 10px;
    border-radius: 8px;
}
</style>
"""
st.markdown(input_style, unsafe_allow_html=True)

# 🧠 Load trained model and scaler
mobile_model = joblib.load("mobile_price_predict.pkl")
mobile_scaler = joblib.load("mobile_price_scaler.pkl")

# 📥 Input layout for user features
col1, col2 = st.columns(2)
with col1:
    weight = st.number_input("Weight (grams)", min_value=0, max_value=800, value=180)
    ppi = st.number_input("Pixels Per Inch", min_value=100, max_value=800, value=300)
    cpu_core = st.number_input("CPU Cores", min_value=1, max_value=16, value=8)
    cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.5, max_value=5.0, value=2.5)
with col2:
    ram = st.number_input("RAM (GB)", min_value=1, max_value=32, value=8)
    RearCam = st.number_input("Rear Camera (MP)", min_value=1, max_value=200, value=48)
    Front_Cam = st.number_input("Front Camera (MP)", min_value=1, max_value=100, value=16)
    thickness = st.number_input("Thickness (mm)", min_value=3.0, max_value=15.0, value=8.0)

# 🚀 Run prediction when button is clicked
if st.button("Predict Price"):
    input_data = np.array([[weight, ppi, cpu_core, cpu_freq, ram,
                            RearCam, Front_Cam, thickness]])
    scaled = mobile_scaler.transform(input_data)
    prediction = mobile_model.predict(scaled)

    st.markdown(
        f"<div style='background-color:#004d00;padding:15px;border-radius:10px;'>"
        f"<h3>📱 Predicted Mobile Price: ₹{prediction[0]:,.2f}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

# ✍️ Footer credit
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: yellow;
        font-size: 16px;
        padding: 10px;
    }
    </style>
    <div class="footer">
        Created by Jyothish
    </div>
    """,
    unsafe_allow_html=True
)
