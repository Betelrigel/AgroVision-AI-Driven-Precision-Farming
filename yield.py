"""
Script to train and save yield prediction models for AgroVision using synthetic data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
data = {
    'crop_type': np.random.choice(['Rice', 'Wheat'], n_samples),
    'month': np.random.choice(['June', 'July', 'August', 'November', 'December'], n_samples),
    'avg_temp_c': np.random.uniform(15, 30, n_samples),
    'rainfall_mm': np.random.uniform(50, 300, n_samples),
    'humidity_percent': np.random.uniform(50, 90, n_samples),
    'soil_ph': np.random.uniform(5.5, 7.5, n_samples),
    'nitrogen_kg_ha': np.random.uniform(50, 150, n_samples),
    'disease_incidence': np.zeros(n_samples),
    'yield_t_ha': np.zeros(n_samples)
}

df = pd.DataFrame(data)

# Assign disease incidence based on crop and weather conditions
for i in range(n_samples):
    if df.loc[i, 'crop_type'] == 'Rice':
        # False smut: Higher risk with high humidity (>80%) and warm temperature (25-30°C)
        if df.loc[i, 'humidity_percent'] > 80 and df.loc[i, 'avg_temp_c'] > 25:
            df.loc[i, 'disease_incidence'] = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            df.loc[i, 'disease_incidence'] = np.random.choice([0, 1], p=[0.8, 0.2])
        # Rice yield: 2-5 t/ha, reduced by disease
        base_yield = np.random.uniform(2, 5)
        df.loc[i, 'yield_t_ha'] = base_yield * (0.7 if df.loc[i, 'disease_incidence'] else 1.0)
    else:
        # Yellow rust: Higher risk with cool temperature (15-20°C) and moderate rainfall
        if df.loc[i, 'avg_temp_c'] < 20 and df.loc[i, 'rainfall_mm'] > 100:
            df.loc[i, 'disease_incidence'] = np.random.choice([0, 1], p=[0.4, 0.6])
        else:
            df.loc[i, 'disease_incidence'] = np.random.choice([0, 1], p=[0.9, 0.1])
        # Wheat yield: 1.5-4 t/ha, reduced by disease
        base_yield = np.random.uniform(1.5, 4)
        df.loc[i, 'yield_t_ha'] = base_yield * (0.65 if df.loc[i, 'disease_incidence'] else 1.0)

# Encode categorical variables
le_crop = LabelEncoder()
le_month = LabelEncoder()
df['crop_type'] = le_crop.fit_transform(df['crop_type'])
df['month'] = le_month.fit_transform(df['month'])

# Features and targets
features = ['crop_type', 'month', 'avg_temp_c', 'rainfall_mm', 'humidity_percent', 'soil_ph', 'nitrogen_kg_ha']
X = df[features]
y_yield = df['yield_t_ha']
y_disease = df['disease_incidence']

# Split data
X_train, X_test, y_yield_train, y_yield_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)
X_train_d, X_test_d, y_disease_train, y_disease_test = train_test_split(X, y_disease, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_d = scaler.fit_transform(X_train_d)
X_test_d = scaler.transform(X_test_d)

# Train yield prediction model
rf_yield = RandomForestRegressor(n_estimators=100, random_state=42)
rf_yield.fit(X_train, y_yield_train)
y_yield_pred = rf_yield.predict(X_test)
yield_rmse = np.sqrt(mean_squared_error(y_yield_test, y_yield_pred))

# Train disease prediction model
rf_disease = RandomForestClassifier(n_estimators=100, random_state=42)
rf_disease.fit(X_train_d, y_disease_train)
y_disease_pred = rf_disease.predict(X_test_d)
disease_accuracy = accuracy_score(y_disease_test, y_disease_pred)
disease_f1 = f1_score(y_disease_test, y_disease_pred)

# Print evaluation metrics
print(f"Yield Prediction RMSE: {yield_rmse:.2f} t/ha")
print(f"Disease Prediction Accuracy: {disease_accuracy:.2f}")
print(f"Disease Prediction F1-Score: {disease_f1:.2f}")

# Feature importance for yield model
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_yield.feature_importances_
}).sort_values(by='importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Yield Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Save synthetic data to CSV
df.to_csv('synthetic_crop_data.csv', index=False)

# Example prediction for a new data point
new_data = pd.DataFrame({
    'crop_type': [le_crop.transform(['Rice'])[0]],
    'month': [le_month.transform(['July'])[0]],
    'avg_temp_c': [28],
    'rainfall_mm': [200],
    'humidity_percent': [85],
    'soil_ph': [6.5],
    'nitrogen_kg_ha': [100]
})
new_data_scaled = scaler.transform(new_data)
predicted_yield = rf_yield.predict(new_data_scaled)[0]
predicted_disease = rf_disease.predict(new_data_scaled)[0]
print(f"Predicted Yield for Rice in July: {predicted_yield:.2f} t/ha")
print(f"Predicted Disease (False Smut): {'Yes' if predicted_disease else 'No'}")

# Save models and objects to .pkl files
os.makedirs('models', exist_ok=True)
joblib.dump(rf_yield, 'models/yield_rf_yield.pkl')
joblib.dump(rf_disease, 'models/yield_rf_disease.pkl')
joblib.dump(le_crop, 'models/yield_le_crop.pkl')
joblib.dump(le_month, 'models/yield_le_month.pkl')
joblib.dump(scaler, 'models/yield_scaler.pkl')

print("Models, encoders, and scaler saved successfully in the 'models' directory.")