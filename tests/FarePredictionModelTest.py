import pickle
import pandas as pd

# ======================
# Load Model
# ======================
with open("src/models/fare_prediction_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# ======================
# New Booking Input
# ======================
input_data = {
    "day_of_week": 2,
    "is_weekend": 0,
    "hour_of_day": 18,
    "city": 3,
    "vehicle_type": 1,
    "ride_distance_km": 12.5,
    "estimated_ride_time_min": 28,
    "traffic_level": 2,
    "weather_condition": 1,
    "base_fare": 50,
    "surge_multiplier": 1.4,
    "customer_gender": 1,
    "customer_age": 32,
    "customer_city": 3,
    "customer_signup_days_ago": 450,
    "preferred_vehicle_type": 1,
    "total_bookings": 24,
    "cancellation_rate": 0.08,
    "avg_customer_rating": 4.6,
    "customer_cancel_flag": 0,
    "driver_age": 41,
    "driver_city": 3,
    "vehicle_type_driver": 1,
    "driver_experience_years": 8,
    "total_assigned_rides": 520,
    "accepted_rides": 495,
    "incomplete_rides_driver": 10,
    "delay_count": 14,
    "acceptance_rate": 0.95,
    "delay_rate": 0.03,
    "avg_driver_rating": 4.7,
    "avg_pickup_delay_min": 3.2,
    "driver_delay_flag": 0,
    "avg_wait_time_min": 4.1,
    "avg_surge_multiplier": 1.2,
    "demand_level": 2,
    "is_holiday": 0,
    "peak_time_flag": 1,
    "season": 1,
    "pickup_location_freq": 0.012,
    "drop_location_freq": 0.010,
    "booking_day": 15,
    "booking_month": 2,
    "booking_year": 2026,
    "booking_hour": 18,
    "fare_per_km": 4.0,
    "fare_per_min": 1.8,
    "surge_demand_interaction": 2.8,
    "rush_hour_flag": 1,
    "long_distance_flag": 0,
    "delay_ratio": 0.02,
    "surge_high_flag": 1,
}

# ======================
# Prediction
# ======================
input_df = pd.DataFrame([input_data])
predicted_fare = pipeline.predict(input_df)[0]

print("Predicted Fare:", round(predicted_fare, 2))
