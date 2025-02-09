import streamlit as st
import pickle
import numpy as np
from PIL import Image
import base64

# Function to set a background image
def set_bg(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Load Model and Vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open("incident_severity_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("tfidf_vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

# Predict Severity Function
def predict_severity(description):
    model, vectorizer = load_model_and_vectorizer()
    description_tfidf = vectorizer.transform([description])
    prediction = model.predict(description_tfidf)
    return prediction[0]

# Set Page Configurations
st.set_page_config(
    page_title="Incident Severity Predictor",
    page_icon="⚠️",
    layout="centered"
)

# Set Background Image
set_bg("safe.jpeg")

# Title and Header
st.markdown(
    """
    <div style="background-color: black; color: white; padding: 20px; border-radius: 5px;">
    <h1 style='text-align: center; color: white;'>🚨 Incident Severity Predictor 🚨</h1>
    </div>
    """,
    unsafe_allow_html=True

)
st.markdown(
    """
    <div style="background-color: black; color: white; padding: 20px; border-radius: 5px;">
    <h3 style='text-align: center; color: white;'>🔍 AI-Powered Safety Analysis for Women & Children</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Slogans and Awareness Quotes
st.markdown(
    """
    <div style="background-color: black; color: white; padding: 20px; border-radius: 5px;">
    🌟 "Your safety matters! Report incidents and raise awareness." <br>
    🛡️ "AI for a safer tomorrow – let's predict, prevent, and protect." <br>
    🔥 "Speak up, stay safe. Together, we build a secure society." 
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")  # Divider Line

# Input Area for Incident Description
user_input = st.text_area(
    "📝 Describe the incident:", 
    placeholder="Example: A working woman faced molestation near a railway station in Mumbai..."
)

# Prediction Button
if st.button("🔍 Predict Severity"):
    if user_input.strip():
        severity = predict_severity(user_input)
        st.markdown(
            f"<h2 style='text-align: center; color: yellow;'>⚠️ Predicted Severity: {severity}</h2>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter a description of the incident.")

# Footer
st.markdown(
    """
    <div style="background-color: black; color: white; padding: 20px; border-radius: 5px;">
    🔔 **Stay Aware, Stay Safe!**
    </div>
    """,
    unsafe_allow_html=True
)
