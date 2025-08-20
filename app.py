import streamlit as st
from PIL import Image
import pandas as pd
import torch
import joblib
# The get_crop_recommendations function is no longer needed here
from utils import preprocess_image, get_weather_api_data, load_disease_model, get_treatment_recommendations, calculate_disease_severity, format_weather_advice, analyze_crop_image
from bot import show_groq_chatbot
from datetime import datetime

# Load pre-trained models and encoders
@st.cache_resource
def load_models():
    rf_yield = joblib.load('models/yield_rf_yield.pkl')
    rf_disease = joblib.load('models/yield_rf_disease.pkl')
    le_crop = joblib.load('models/yield_le_crop.pkl')
    le_month = joblib.load('models/yield_le_month.pkl')
    scaler = joblib.load('models/yield_scaler.pkl')
    return rf_yield, rf_disease, le_crop, le_month, scaler

rf_yield, rf_disease, le_crop, le_month, scaler = load_models()

def show_disease_detection():
    """Simple disease detection feature"""
    st.markdown("### 🔍 Crop Disease Detection")
    st.markdown("Upload an image of your crop to detect diseases")
    
    crop_type = st.selectbox("Select your crop:", ["Wheat", "Rice"])
    
    uploaded_file = st.file_uploader("Choose crop image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image..."):
                result = analyze_crop_image(image, crop_type)
                
                if result['disease_detected']:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>⚠️ Disease Detected: {result['disease_name']}</h4>
                        <p><strong>Confidence:</strong> {result['confidence']}%</p>
                        <p><strong>Description:</strong> {result['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### 💊 Treatment Recommendations:")
                    for i, treatment in enumerate(result['treatments'], 1):
                        st.markdown(f"{i}. {treatment}")
                else:
                    st.markdown(f"""
                    <div class="result-box">
                        <h4>✅ Healthy Crop Detected</h4>
                        <p>Your {crop_type.lower()} crop appears healthy!</p>
                        <p><strong>Confidence:</strong> {result['confidence']}%</p>
                    </div>
                    """, unsafe_allow_html=True)

def show_weather_tips():
    """Weather-based recommendations using API"""
    st.markdown("### 🌤️ Weather-Based Farming Tips")
    st.markdown("Get farming advice based on current weather conditions in Hamirpur")
    
    API_KEY = 'your api key here'

    @st.cache_data(ttl=3600)
    def get_weather_data():
        weather_data = get_weather_api_data(api_key=API_KEY)
        if weather_data:
            weather_data['last_updated'] = datetime.now().strftime('%I:%M %p IST on %B %d, %Y')
            return weather_data
        else:
            st.error("Failed to fetch weather data.")
            return None

    weather_data = get_weather_data()
    
    if weather_data:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Temperature", f"{weather_data['temperature']}°C")
        with col2: st.metric("Humidity", f"{weather_data['humidity']}%")
        with col3: st.metric("Rainfall", f"{weather_data['rainfall']}mm")
        st.write(f"**Condition:** {weather_data['condition']}")
        st.write(f"**Last Updated:** {weather_data['last_updated']}")
        
        st.markdown("#### 🌾 Today's Farming Recommendations:")
        recommendations = format_weather_advice(weather_data)
        for rec in recommendations:
            st.markdown(f"""
            <div class="feature-box">
                {rec}
            </div>
            """, unsafe_allow_html=True)

# --- The show_recommendations() function has been completely removed ---

def show_yield_prediction():
    """Yield prediction feature"""
    st.markdown("### 🌾 Crop Yield Prediction")
    st.markdown("Enter crop and weather data to predict yield and disease incidence")
    
    crop_type = st.selectbox("Crop Type:", ["Rice", "Wheat"])
    month = st.selectbox("Month:", ["June", "July", "August", "November", "December"])
    avg_temp = st.number_input("Average Temperature (°C):", min_value=15.0, max_value=30.0, value=22.0)
    rainfall = st.number_input("Rainfall (mm):", min_value=50.0, max_value=300.0, value=150.0)
    humidity = st.number_input("Humidity (%):", min_value=50.0, max_value=90.0, value=65.0)
    soil_ph = st.number_input("Soil pH:", min_value=5.5, max_value=7.5, value=6.5)
    nitrogen = st.number_input("Nitrogen (kg/ha):", min_value=50.0, max_value=150.0, value=100.0)
    
    if st.button("Predict Yield"):
        new_data = pd.DataFrame({
            'crop_type': [le_crop.transform([crop_type])[0]],
            'month': [le_month.transform([month])[0]],
            'avg_temp_c': [avg_temp],
            'rainfall_mm': [rainfall],
            'humidity_percent': [humidity],
            'soil_ph': [soil_ph],
            'nitrogen_kg_ha': [nitrogen]
        })
        new_data_scaled = scaler.transform(new_data)
        predicted_yield = rf_yield.predict(new_data_scaled)[0]
        predicted_disease = rf_disease.predict(new_data_scaled)[0]
        st.markdown(f"""
        <div class="result-box">
            <h4>Prediction Results</h4>
            <p><strong>Predicted Yield:</strong> {predicted_yield:.2f} t/ha</p>
            <p><strong>Disease Incidence:</strong> {'Yes' if predicted_disease else 'No'}</p>
        </div>
        """, unsafe_allow_html=True)
        if predicted_disease:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ Disease Risk Detected</h4>
                <p>Monitor your {crop_type.lower()} crop closely and consider treatments.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box">
                <p>Low disease risk detected for {crop_type.lower()}.</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-title">🌾 AgroVision - Smart Farming Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">For Hamirpur, Himachal Pradesh Farmers</p>', unsafe_allow_html=True)
    
    # Add CSS for styling
    st.markdown("""
    <style>
    .feature-box, .result-box, .warning-box {
        border: 1px solid #ccc;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        border-color: #ff4d4d;
        background-color: #ffe6e6;
    }
    .result-box {
        border-color: #4CAF50;
        background-color: #e6ffe6;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 16px;
        color: #666;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # --- UPDATED: Removed the Recommendations tab and adjusted the remaining ones ---
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Disease Detection", "🌤️ Weather Tips", "💬 Chatbot", "🌾 Yield Prediction"])
    
    with tab1:
        show_disease_detection()
    
    with tab2:
        show_weather_tips()
    
    with tab3: # Was tab4
        show_groq_chatbot()
    
    with tab4: # Was tab5
        show_yield_prediction()

if __name__ == "__main__":

    main()
