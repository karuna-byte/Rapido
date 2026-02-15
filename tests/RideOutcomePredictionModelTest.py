import pickle
import pandas as pd

# Load model
with open("src/models/ride_outcome_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature list
with open("src/models/ride_outcome_features.pkl", "rb") as f:
    FEATURES_STRICT = pickle.load(f)

print("Model loaded successfully")

# SEND INPUT DATA (SINGLE BOOKING)

input_data = {
    "day_of_week": 2,
    "is_weekend": 0,
    "hour_of_day": 18,
    "booking_day": 15,
    "booking_month": 2,
    "booking_year": 2026,
    "booking_hour": 18,
    "is_holiday": 0,
    "peak_time_flag": 1,
    "season": 1,
    "ride_distance_km": 12.5,
    "estimated_ride_time_min": 28,
    "traffic_level": 2,
    "weather_condition": 1,
    "base_fare": 50,
    "surge_multiplier": 1.4,
    "booking_value": 210,
    "fare_per_km": 50 / (12.5 + 1),
    "fare_per_min": 50 / (28 + 1),
    "rush_hour_flag": 1,
    "long_distance_flag": 0,
    "surge_high_flag": 0,
}


# CONVERT TO DATAFRAME (ORDER SAFE)
input_df = pd.DataFrame([input_data])

# Enforce correct column order
input_df = input_df[FEATURES_STRICT]

# PREDICT RIDE OUTCOME
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0]

status_map = {0: "Cancelled", 1: "Completed", 2: "Incomplete"}

print("Predicted Ride Outcome:", status_map[prediction])
print("Prediction Probabilities:", proba)
