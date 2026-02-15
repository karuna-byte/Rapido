import joblib
import pandas as pd
import numpy as np

# ======================================================
# 1. LOAD MODEL
# ======================================================

model = joblib.load("src/models/driver_delay_prediction_model.pkl")

# ======================================================
# 2. FEATURE LIST (MUST MATCH TRAINING)
# ======================================================

features = [
    "driver_experience_years",
    "total_assigned_rides",
    "accepted_rides",
    "avg_driver_rating",
    "driver_acceptance_rate",
    "exp_rating",
    "ride_distance_km",
    "estimated_ride_time_min",
    "distance_time_ratio",
    "surge_multiplier",
    "surge_demand_interaction",
    "hour_of_day",
    "is_weekend",
    "traffic_level",
    "weather_condition",
    "avg_wait_time_min",
    "avg_surge_multiplier",
    "demand_level",
    "peak_time_flag",
    "is_holiday",
    "season",
]

# ======================================================
# 3. INPUT DATA
# ======================================================

input_data = {
    "driver_experience_years": 4,
    "total_assigned_rides": 500,
    "accepted_rides": 420,
    "avg_driver_rating": 4.6,
    "ride_distance_km": 12,
    "estimated_ride_time_min": 28,
    "surge_multiplier": 1.8,
    "hour_of_day": 18,
    "is_weekend": 0,
    "traffic_level": 3,
    "weather_condition": "Clear",
    "avg_wait_time_min": 7,
    "avg_surge_multiplier": 1.5,
    "demand_level": 4,
    "peak_time_flag": 1,
    "is_holiday": 0,
    "season": "Winter",
}

df_input = pd.DataFrame([input_data])

# ======================================================
# 4. RECREATE DERIVED FEATURES
# ======================================================

df_input["driver_acceptance_rate"] = df_input["accepted_rides"] / (
    df_input["total_assigned_rides"] + 1
)

df_input["distance_time_ratio"] = df_input["ride_distance_km"] / (
    df_input["estimated_ride_time_min"] + 1
)

df_input["surge_demand_interaction"] = (
    df_input["surge_multiplier"] * df_input["demand_level"]
)

df_input["exp_rating"] = (
    df_input["driver_experience_years"] * df_input["avg_driver_rating"]
)

# Keep same feature order
df_input = df_input[features]

# ======================================================
# 🔥 CRITICAL FIX
# Convert ALL object columns to numeric codes
# This removes pandas categorical dependency
# ======================================================

for col in df_input.select_dtypes(include="object").columns:
    df_input[col] = df_input[col].astype("category").cat.codes

# Convert to numpy array (BYPASS pandas validation)
X_input = df_input.values

# ======================================================
# 5. PREDICT
# ======================================================

probability = model.predict_proba(X_input)[0][1]

threshold = 0.55
prediction = 1 if probability >= threshold else 0

print("\nDelay Probability:", round(probability, 4))
print("Prediction:", prediction)

if prediction == 1:
    print("⚠️ Driver Likely To Be Delayed")
else:
    print("✅ Driver Likely On Time")
