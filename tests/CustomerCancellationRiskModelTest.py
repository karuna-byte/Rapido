import pickle
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD MODEL
# ==========================================

with open("src/models/customer_cancellation_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")

# Get training feature names
feature_columns = model.feature_names_in_

# ==========================================
# 2. CREATE INPUT DATA
# ==========================================

new_data = {
    "customer_id": 1012,
    "driver_id": 455,
    "city": 2,
    "pickup_location": 10,
    "vehicle_type": 1,
    "ride_distance_km": 8.5,
    "estimated_ride_time_min": 20,
    "booking_value": 240,
    "surge_multiplier": 1.2,
    "hour_of_day": 18,
    "day_of_week": 5,
    "is_weekend": 1,
    "booking_month": 2,
    "booking_day": 13,
    "booking_hour": 18,
    "fare_per_km": 240 / (8.5 + 1),
    "fare_per_min": 240 / (20 + 1),
    "long_distance_flag": 0,
    "high_surge_flag": 0,
}

input_df = pd.DataFrame([new_data])

# ==========================================
# 3. ADD MISSING COLUMNS AUTOMATICALLY
# ==========================================

for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns exactly as training
input_df = input_df[feature_columns]

# ==========================================
# 4. PREDICT
# ==========================================

prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

print("Prediction (0=No Cancel, 1=Cancel):", prediction)
print("Cancellation Probability:", round(probability, 4))
