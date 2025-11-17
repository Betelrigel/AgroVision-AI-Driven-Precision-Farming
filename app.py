import streamlit as st
from PIL import Image
import pandas as pd
import joblib
from datetime import datetime

from utils import (
    preprocess_image, get_weather_api_data, load_disease_model,
    get_treatment_recommendations, calculate_disease_severity,
    format_weather_advice, analyze_crop_image
)
from bot import show_groq_chatbot

# -----------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------
@st.cache_resource
def load_models():
    rf_yield = joblib.load('models/yield_rf_yield.pkl')
    rf_disease = joblib.load('models/yield_rf_disease.pkl')
    le_crop = joblib.load('models/yield_le_crop.pkl')
    le_month = joblib.load('models/yield_le_month.pkl')
    scaler = joblib.load('models/yield_scaler.pkl')
    return rf_yield, rf_disease, le_crop, le_month, scaler

rf_yield, rf_disease, le_crop, le_month, scaler = load_models()

# -----------------------------------------------------
# MODERN MINIMAL UI
# -----------------------------------------------------
st.markdown("""
<style>
body {
    font-family: 'Inter', sans-serif;
    color: #222;
}

/* Glassmorphism Card */
.card {
    background: rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
    padding: 1.2rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.4);
    transition: 0.2s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 22px rgba(0,0,0,0.10);
}

/* Title */
.app-title {
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #2E7D32, #4CAF50);
    -webkit-background-clip: text;
    color: transparent;
    margin-top: -10px;
}

/* Subtitle */
.app-subtitle {
    text-align: center;
    font-size: 1.05rem;
    margin-top: -12px;
    color: #555;
}

/* Divider */
.divider {
    height: 2px;
    width: 60%;
    margin: 16px auto;
    background: linear-gradient(90deg, #4CAF50, #2E7D32);
    border-radius: 10px;
}

/* Metric Cards */
.metric-card {
    padding: 1rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #F3FFF8, #E8FFEF);
    text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2E7D32, #4CAF50);
    color: white;
    padding: 10px 22px;
    border-radius: 10px;
    font-weight: 600;
    transition: 0.2s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #256628, #43A047);
}

/* Result Boxes */
.result-box {
    background: #E9FFF0;
    border-left: 4px solid #2E7D32;
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
}

.warning-box {
    background: #FFF2F2;
    border-left: 4px solid #C62828;
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
}

.feature-box {
    background: #F4FFF9;
    border-left: 4px solid #4CAF50;
    border-radius: 10px;
    padding: 0.8rem;
    margin: .6rem 0;
}

/* Tabs */
.stTabs [role="tab"] {
    padding: 10px 22px;
    font-size: 1.05rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------
# DISEASE DETECTION
# -----------------------------------------------------
def show_disease_detection():
    st.markdown("### Disease Detection\n### रोग पहचान")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    crop = st.selectbox("Select Crop\nफसल चुनें", ["Wheat\nगेहूं", "Rice\nधान"])
    uploaded = st.file_uploader("Upload Crop Image\nफ़सल की छवि अपलोड करें", ["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image\nअपलोड की गई छवि", use_column_width=True)

        if st.button("Analyze Image\nछवि विश्लेषण करें"):
            with st.spinner("Analyzing image...\nछवि का विश्लेषण हो रहा है..."):
                # Extract English crop name (remove Hindi if present)
                crop_name = crop.split('\n')[0] if '\n' in crop else crop
                result = analyze_crop_image(img, crop_name)

            if result["disease_detected"]:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>{result['disease_name']}</h4>
                    <p><strong>Confidence:</strong> {result['confidence']}%</p>
                    <p>{result['description']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Treatment Recommendations\n#### उपचार सिफारिशें")
                for t in result["treatments"]:
                    st.markdown(f"- {t}")

            else:
                st.markdown(f"""
                <div class="result-box">
                    Healthy Crop — No disease detected<br>स्वस्थ फ़सल — कोई रोग नहीं पाया गया  
                    <br><strong>Confidence:</strong> {result['confidence']}%<br><strong>विश्वसनीयता:</strong> {result['confidence']}%
                </div>
                """, unsafe_allow_html=True)


# -----------------------------------------------------
# WEATHER TIPS
# -----------------------------------------------------
def show_weather_tips():
    st.markdown("### Weather Insights\n### मौसम जानकारी")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    API_KEY = "b11a4b41404cc2571a14597947f4c816"

    @st.cache_data(ttl=3600)
    def fetch_weather():
        data = get_weather_api_data(api_key=API_KEY)
        if data:
            data['last_updated'] = datetime.now().strftime("%I:%M %p • %d %b %Y")
            return data
        return None

    weather = fetch_weather()
    if not weather:
        st.error("Couldn't fetch weather data\nमौसम डेटा प्राप्त नहीं कर पाया")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>{weather['temperature']}°C</h3><p>Temperature<br>तापमान</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>{weather['humidity']}%</h3><p>Humidity<br>आर्द्रता</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>{weather['rainfall']} mm</h3><p>Rainfall<br>वर्षा</p></div>", unsafe_allow_html=True)

    st.write(f"**Condition:** {weather['condition']}")
    st.write(f"**स्थिति:** {weather['condition']}")
    st.caption(f"Last Updated: {weather['last_updated']}")
    st.caption(f"अंतिम अपडेट: {weather['last_updated']}")

    st.markdown("#### Recommended Actions\n#### अनुशंसित क्रियाएँ")
    for rec in format_weather_advice(weather):
        st.markdown(f"<div class='feature-box'>{rec}</div>", unsafe_allow_html=True)


# -----------------------------------------------------
# YIELD PREDICTION
# -----------------------------------------------------
def show_yield_prediction():
    st.markdown("### Yield Prediction\n### उपज भविष्यवाणी")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    crop = st.selectbox("Crop\nफ़सल", ["Rice\nधान", "Wheat\nगेहूं"])
    month = st.selectbox("Month\nमाह", ["June", "July", "August", "November", "December"])

    temp = st.number_input("Avg Temperature (°C)\nऔसत तापमान (°C)", 15.0, 30.0, 22.0)
    rainfall = st.number_input("Rainfall (mm)\nवर्षा (मिमी)", 50.0, 300.0, 150.0)
    humidity = st.number_input("Humidity (%)\nआर्द्रता (%)", 50.0, 90.0, 65.0)
    pH = st.number_input("Soil pH\nमृदा pH", 5.5, 7.5, 6.5)
    nitrogen = st.number_input("Nitrogen (kg/ha)\nनाइट्रोजन (kg/ha)", 50.0, 150.0, 100.0)

    if st.button("Predict Yield\nउपज अनुमानित करें"):
        # Extract English crop name (remove Hindi if present)
        crop_name = crop.split('\n')[0] if '\n' in crop else crop
        df = pd.DataFrame({
            "crop_type": [le_crop.transform([crop_name])[0]],
            "month": [le_month.transform([month])[0]],
            "avg_temp_c": [temp],
            "rainfall_mm": [rainfall],
            "humidity_percent": [humidity],
            "soil_ph": [pH],
            "nitrogen_kg_ha": [nitrogen]
        })

        scaled = scaler.transform(df)
        pred_yield = rf_yield.predict(scaled)[0]
        disease_risk = rf_disease.predict(scaled)[0]

        st.markdown(f"""
        <div class="result-box">
            <h4>Prediction Results<br>अनुमान परिणाम</h4>
            <p><strong>Expected Yield:</strong> {pred_yield:.2f} t/ha<br><strong>अनुमानित उपज:</strong> {pred_yield:.2f} t/ha</p>
            <p><strong>Disease Risk:</strong> {"High" if disease_risk else "Low"}<br><strong>रोग जोखिम:</strong> {"उच्च" if disease_risk else "कम"}</p>
        </div>
        """, unsafe_allow_html=True)

        if disease_risk:
            st.markdown("""
            <div class="warning-box">
            Crop conditions indicate elevated disease risk. Monitor closely.<br>फ़सल की स्थिति में रोग का जोखिम बढ़ा हुआ है। ध्यान से देखें।
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------
# MAIN APP
# -----------------------------------------------------
def main():
    st.markdown('<p class="app-title">AgroVision<br>एग्रोविजन</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">AI-Powered Precision Farming<br>एआई-संचालित प्रिसिजन खेती</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Disease Detection",
        "Weather Insights",
        "AI Chatbot",
        "Yield Prediction"
    ])

    with tab1:
        show_disease_detection()
    with tab2:
        show_weather_tips()
    with tab3:
        show_groq_chatbot()
    with tab4:
        show_yield_prediction()


if __name__ == "__main__":
    main()
