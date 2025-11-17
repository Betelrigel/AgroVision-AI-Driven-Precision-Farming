# utils.py
"""
Simple utility functions for AgroVision app
"""

import cv2
import numpy as np
from PIL import Image
import requests
import joblib
import os

# NOTE: Imports for heavy ML libraries (torch, torchvision, tensorflow) are
# performed lazily inside functions that need them. This avoids loading
# large native DLLs at Streamlit startup and prevents DLL init failures
# in some Windows process contexts. Import errors are re-raised with
# a clearer message pointing to common fixes.

def preprocess_image(image, target_size=(224, 224)):
    """
    Simple image preprocessing for PYTORCH disease detection models
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)

    # Resize image
    img_resized = cv2.resize(img_array, target_size)

    # Normalize pixel values
    img_normalized = img_resized / 255.0

    # Convert to tensor for PyTorch models (import lazily)
    try:
        import torch
    except Exception as e:
        raise ImportError(
            "Failed to import torch while preprocessing image.\n"
            "छवि प्रीप्रोसेसिंग के दौरान torch import विफल।\n"
            "If you are running on Windows, ensure you launched Streamlit from the project's venv, "
            "and that the appropriate Visual C++ redistributable is installed.\n"
            "विंडोज पर चल रहे हैं तो सुनिश्चित करें कि आपने प्रोजेक्ट के venv से स्ट्रीमलिट लॉन्च किया है।\n"
            f"Original error: {e}"
        ) from e

    img_tensor = torch.tensor(img_normalized).permute(2, 0, 1).float().unsqueeze(0)
    return img_tensor

def get_weather_api_data(api_key=None):
    """
    Fetch weather data from API
    """
    hamirpur_coords = {
        'lat': 31.6839,
        'lon': 76.5217
    }
    
    if api_key:
        # Using OpenWeatherMap API
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={hamirpur_coords['lat']}&lon={hamirpur_coords['lon']}&appid={api_key}&units=metric"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes
            data = response.json()
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'rainfall': data.get('rain', {}).get('1h', 0) or 0,
                'wind_speed': data['wind']['speed'],
                'condition': data['weather'][0]['description']
            }
        except requests.exceptions.RequestException:
             # Fallback to mock data if API fails
            weather_data = {'temperature': 22, 'humidity': 65, 'rainfall': 0, 'wind_speed': 5, 'condition': 'Partly Cloudy (API Error)'}
    else:
        # Mock weather data
        weather_data = {'temperature': 22, 'humidity': 65, 'rainfall': 0, 'wind_speed': 5, 'condition': 'Partly Cloudy'}
    
    return weather_data

def load_disease_model(crop_type):
    """
    Load pre-trained disease detection model based on the crop type.
    Handles both PyTorch (.pkl) and TensorFlow/Keras (.h5) models.
    """
    model_paths = {
        'wheat': 'models/wheat_yellow_rust_model.pkl',
        'rice': 'models/rice_disease_model.h5'
    }
    model_path = model_paths.get(crop_type.lower())

    if not model_path:
        raise ValueError(f"No model path defined for crop: {crop_type} / फ़सल के लिए कोई मॉडल पथ परिभाषित नहीं: {crop_type}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file was not found at: {model_path} / मॉडल फ़ाइल यहाँ नहीं पाई गई: {model_path}")

    if crop_type.lower() == 'wheat':
        # --- Load PyTorch Model (lazy imports) ---
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
        except Exception as e:
            raise ImportError(
                "Failed to import PyTorch or torchvision when loading the wheat model.\n"
                "गेहूं मॉडल लोड करते समय PyTorch या torchvision import विफल।\n"
                "Ensure you installed a compatible CPU/GPU build in the venv and launched Streamlit from that venv.\n"
                "सुनिश्चित करें कि आपने venv में संगत CPU/GPU बिल्ड स्थापित किया है।\n"
                f"Original error: {e}"
            ) from e

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, 6)
        model.load_state_dict(joblib.load(model_path))
        model.eval()
        return model.to(device)

    elif crop_type.lower() == 'rice':
        # --- Load TensorFlow/Keras Model (lazy import) ---
        try:
            import tensorflow as tf
        except Exception as e:
            raise ImportError(
                "Failed to import TensorFlow when loading the rice model.\n"
                "चावल मॉडल लोड करते समय TensorFlow import विफल।\n"
                "On Windows this often means the TensorFlow binary is incompatible with the installed Python, "
                "or missing system dependencies. See https://www.tensorflow.org/install/errors.\n"
                "विंडोज पर इसका मतलब अक्सर यह है कि TensorFlow बाइनरी इंस्टॉल किए गए Python के साथ असंगत है।\n"
                f"Original error: {e}"
            ) from e

        model = tf.keras.models.load_model(model_path)
        return model
        
def load_yield_models():
    """
    Load pre-trained yield prediction models and related objects
    """
    try:
        rf_yield = joblib.load('models/yield_rf_yield.pkl')
        rf_disease = joblib.load('models/yield_rf_disease.pkl')
        le_crop = joblib.load('models/yield_le_crop.pkl')
        le_month = joblib.load('models/yield_le_month.pkl')
        scaler = joblib.load('models/yield_scaler.pkl')
        return rf_yield, rf_disease, le_crop, le_month, scaler
    except FileNotFoundError:
        raise Exception("Yield prediction models not found. Please train and save them first.")

def calculate_disease_severity(image, mask=None):
    """
    Calculate disease severity from image
    """
    if mask is not None:
        affected_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        severity = (affected_pixels / total_pixels) * 100
    else:
        import random
        severity = random.randint(10, 80)
    
    if severity < 25: return "Low"
    elif severity < 50: return "Medium"
    else: return "High"

def get_treatment_recommendations(crop, disease, severity):
    """
    Get treatment recommendations based on crop, disease, and severity
    """
    treatments = {
        'wheat': {
            'yellow_rust': {
                'Low': ["Monitor field conditions closely", "Apply preventive fungicide if weather favors disease", "Ensure proper field drainage"],
                'Medium': ["Apply systemic fungicide (Propiconazole)", "Spray every 15 days until control", "Remove infected plant debris"],
                'High': ["Immediate fungicide application required", "Use combination of systemic and contact fungicides", "Consider crop loss assessment"]
            }
        },
        'rice': {
            'false_smut': {
                'Low': ["Monitor during flowering stage", "Ensure proper field ventilation", "Use balanced fertilization"],
                'Medium': ["Apply copper-based fungicides", "Improve water management", "Remove affected panicles"],
                'High': ["Intensive fungicide treatment required", "Harvest early if possible", "Destroy infected plant material"]
            },
            'bacterial_blight_disease': {
                'Low': ["Monitor field for early signs", "Ensure proper water drainage", "Use resistant varieties"],
                'Medium': ["Apply bactericides (Copper Oxychloride)", "Avoid overhead irrigation", "Remove infected leaves"],
                'High': ["Immediate bactericide application", "Use integrated pest management", "Assess crop viability"]
            },
            'blast_disease': {
                'Low': ["Monitor during humid conditions", "Use balanced nitrogen", "Improve field aeration"],
                'Medium': ["Apply Tricyclazole fungicide", "Avoid excessive nitrogen", "Remove infected parts"],
                'High': ["Intensive fungicide treatment", "Consider early harvest", "Destroy infected debris"]
            },
            'brown_spot_disease': {
                'Low': ["Monitor during wet weather", "Ensure proper drainage", "Use resistant varieties"],
                'Medium': ["Apply Mancozeb fungicide", "Improve field sanitation", "Reduce water stagnation"],
                'High': ["Immediate fungicide application", "Integrated disease management", "Evaluate crop loss"]
            }
        }
    }
    return treatments.get(crop, {}).get(disease, {}).get(severity, ["No specific recommendation found."])

def format_weather_advice(weather_data):
    """
    Format weather data into actionable farming advice
    """
    advice = []
    temp = weather_data.get('temperature', 0)
    humidity = weather_data.get('humidity', 0)
    rainfall = weather_data.get('rainfall', 0)
    
    if temp > 30: advice.append("🌡️ High temperature: Plan irrigation for early morning or evening")
    elif temp < 10: advice.append("🌡️ Low temperature: Protect crops from frost damage")
    
    if humidity > 80: advice.append("💧 High humidity: Monitor for fungal diseases")
    elif humidity < 40: advice.append("💧 Low humidity: Increase irrigation frequency")
    
    if rainfall > 10: advice.append("🌧️ Good rainfall: Delay irrigation, check for waterlogging")
    elif rainfall == 0: advice.append("☀️ No rainfall: Plan irrigation schedule")
    
    return advice

def validate_yield_input(avg_temp_c, rainfall_mm, humidity_percent, soil_ph, nitrogen_kg_ha):
    """
    Validate input data for yield prediction
    """
    if not (15 <= avg_temp_c <= 30): raise ValueError("Temperature must be between 15°C and 30°C")
    if not (50 <= rainfall_mm <= 300): raise ValueError("Rainfall must be between 50mm and 300mm")
    if not (50 <= humidity_percent <= 90): raise ValueError("Humidity must be between 50% and 90%")
    if not (5.5 <= soil_ph <= 7.5): raise ValueError("Soil pH must be between 5.5 and 7.5")
    if not (50 <= nitrogen_kg_ha <= 150): raise ValueError("Nitrogen must be between 50kg/ha and 150kg/ha")
    return True

def get_crop_recommendations(crop, season):
    """
    Provide general farming recommendations based on crop and season.
    This version ensures it ALWAYS returns a dictionary to prevent 'NoneType' errors.
    """
    recommendations = {
        'wheat': {
            'Planting': ["Sow seeds at a depth of 5-6 cm", "Ensure row spacing of 20-22 cm", "Use certified seeds for better yield"],
            'Irrigation': ["Irrigate 4-5 times during the season", "First irrigation 20-25 days after sowing", "Avoid waterlogging in heavy soils"],
            'Fertilization': ["Apply 120 kg N, 60 kg P, 40 kg K per hectare", "Use urea in split doses", "Test soil for nutrient deficiencies"]
        },
        'rice': {
            'Planting': ["Transplant seedlings at 20-25 days old", "Maintain 20x15 cm spacing", "Use hybrid varieties for higher yield"],
            'Irrigation': ["Keep field flooded during early growth", "Maintain 5 cm water level during tillering", "Drain water 10 days before harvest"],
            'Fertilization': ["Apply 100 kg N, 50 kg P, 50 kg K per hectare", "Use nitrogen in 3 split doses", "Incorporate organic manure"]
        }
    }
    
    # Adjust recommendations based on season
    if season == "Kharif (June-Nov)":
        if crop.lower() == 'rice':
            return recommendations.get('rice', {})
        else:
            return {'General': ['Wheat is typically a Rabi season crop. Consider planting rice or other Kharif crops.']}
    elif season == "Rabi (Dec-May)":
        if crop.lower() == 'wheat':
            return recommendations.get('wheat', {})
        else:
            return {'General': ['Rice is typically a Kharif season crop. Consider planting wheat or other Rabi crops.']}
    
    # --- THIS IS THE FIX for the 'NoneType' error ---
    # Add a fallback return statement to handle any other season.
    # It returns recommendations for the selected crop, or an empty dictionary if the crop isn't found.
    return recommendations.get(crop.lower(), {})

def analyze_crop_image(image, crop_type):
    """
    Analyze crop image for disease detection using the appropriate model.
    """
    model = load_disease_model(crop_type.lower())

    # --- Prediction logic now depends on the crop type and its model framework ---
    if crop_type.lower() == 'wheat':
        # --- Use PyTorch for the wheat model (lazy import) ---
        try:
            import torch
        except Exception as e:
            raise ImportError(
                "Failed to import torch while analyzing image.\n"
                "छवि विश्लेषण के दौरान torch import विफल।\n"
                "Try running Streamlit from the project venv and ensure the PyTorch build is compatible with your system.\n"
                "प्रोजेक्ट venv से Streamlit चलाने का प्रयास करें।\n"
                f"Original error: {e}"
            ) from e

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processed_image = preprocess_image(image)
        with torch.no_grad():
            output = model(processed_image.to(device))
            probabilities = torch.softmax(output, 1)
            confidence, predicted_idx_tensor = torch.max(probabilities, 1)
            predicted_idx = predicted_idx_tensor.item()
            confidence = confidence.item() * 100

    elif crop_type.lower() == 'rice':
        # --- Use TensorFlow/Keras for the rice model ---
        img_array = np.array(image.resize((224, 224)))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
    else:
        raise ValueError("Unsupported crop type for analysis / विश्लेषण के लिए असमर्थित फ़सल प्रकार")

    # --- Common logic to format the result ---
    severity = calculate_disease_severity(image)
    disease_classes = {
        'wheat': ['0', 'MR', 'MRMS', 'MS', 'R', 'S'],
        'rice': ['Bacterial Blight Disease', 'Blast Disease', 'Brown Spot Disease', 'False Smut Disease']
    }
    disease_name = disease_classes[crop_type.lower()][predicted_idx]
    disease_detected = disease_name != '0' if crop_type.lower() == 'wheat' else True

    return {
        'disease_detected': disease_detected,
        'disease_name': disease_name if disease_detected else 'None / कोई नहीं',
        'confidence': round(confidence, 2),
        'description': f"{disease_name.replace('_', ' ').title()} detected / {disease_name.replace('_', ' ').title()} पाया गया" if disease_detected else 'No disease detected / कोई रोग नहीं पाया गया',
        'treatments': get_treatment_recommendations(crop_type.lower(), disease_name.lower().replace(' ', '_'), severity) if disease_detected else []
    }