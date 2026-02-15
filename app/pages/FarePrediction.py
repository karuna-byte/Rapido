import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt


# ======================
# Page Config
# ======================
def fare_prediction_page():
    st.set_page_config(page_title="Fare Prediction App", layout="wide")

    st.title("🚖 Ride Fare Prediction System")

    # ======================
    # Load Model
    # ======================
    @st.cache_resource
    def load_model():
        with open("src/models/fare_prediction_pipeline.pkl", "rb") as f:
            model = pickle.load(f)
        return model

    pipeline = load_model()

    # ======================
    # Sidebar Inputs
    # ======================
    st.sidebar.header("Enter Booking Details")

    input_data = {
        "day_of_week": st.sidebar.selectbox("Day of Week (0=Mon)", list(range(7))),
        "is_weekend": st.sidebar.selectbox("Is Weekend", [0, 1]),
        "hour_of_day": st.sidebar.slider("Hour of Day", 0, 23, 18),
        "city": st.sidebar.number_input("City Code", 0, 10, 3),
        "vehicle_type": st.sidebar.number_input("Vehicle Type", 0, 5, 1),
        "ride_distance_km": st.sidebar.number_input(
            "Ride Distance (km)", 0.0, 100.0, 12.5
        ),
        "estimated_ride_time_min": st.sidebar.number_input(
            "Estimated Ride Time (min)", 0.0, 180.0, 28.0
        ),
        "traffic_level": st.sidebar.selectbox("Traffic Level", [0, 1, 2, 3]),
        "weather_condition": st.sidebar.selectbox("Weather Condition", [0, 1, 2]),
        "base_fare": st.sidebar.number_input("Base Fare", 0.0, 500.0, 50.0),
        "surge_multiplier": st.sidebar.number_input("Surge Multiplier", 1.0, 5.0, 1.4),
        "customer_gender": st.sidebar.selectbox("Customer Gender", [0, 1]),
        "customer_age": st.sidebar.number_input("Customer Age", 18, 80, 32),
        "customer_city": st.sidebar.number_input("Customer City", 0, 10, 3),
        "customer_signup_days_ago": st.sidebar.number_input(
            "Signup Days Ago", 0, 5000, 450
        ),
        "preferred_vehicle_type": st.sidebar.number_input(
            "Preferred Vehicle Type", 0, 5, 1
        ),
        "total_bookings": st.sidebar.number_input("Total Bookings", 0, 1000, 24),
        "cancellation_rate": st.sidebar.number_input(
            "Cancellation Rate", 0.0, 1.0, 0.08
        ),
        "avg_customer_rating": st.sidebar.number_input(
            "Avg Customer Rating", 0.0, 5.0, 4.6
        ),
        "customer_cancel_flag": st.sidebar.selectbox("Customer Cancel Flag", [0, 1]),
        "driver_age": st.sidebar.number_input("Driver Age", 18, 80, 41),
        "driver_city": st.sidebar.number_input("Driver City", 0, 10, 3),
        "vehicle_type_driver": st.sidebar.number_input("Driver Vehicle Type", 0, 5, 1),
        "driver_experience_years": st.sidebar.number_input(
            "Driver Experience (Years)", 0, 40, 8
        ),
        "total_assigned_rides": st.sidebar.number_input(
            "Total Assigned Rides", 0, 10000, 520
        ),
        "accepted_rides": st.sidebar.number_input("Accepted Rides", 0, 10000, 495),
        "incomplete_rides_driver": st.sidebar.number_input(
            "Incomplete Rides", 0, 1000, 10
        ),
        "delay_count": st.sidebar.number_input("Delay Count", 0, 1000, 14),
        "acceptance_rate": st.sidebar.number_input("Acceptance Rate", 0.0, 1.0, 0.95),
        "delay_rate": st.sidebar.number_input("Delay Rate", 0.0, 1.0, 0.03),
        "avg_driver_rating": st.sidebar.number_input(
            "Avg Driver Rating", 0.0, 5.0, 4.7
        ),
        "avg_pickup_delay_min": st.sidebar.number_input(
            "Avg Pickup Delay (min)", 0.0, 60.0, 3.2
        ),
        "driver_delay_flag": st.sidebar.selectbox("Driver Delay Flag", [0, 1]),
        "avg_wait_time_min": st.sidebar.number_input(
            "Avg Wait Time (min)", 0.0, 60.0, 4.1
        ),
        "avg_surge_multiplier": st.sidebar.number_input(
            "Avg Surge Multiplier", 1.0, 5.0, 1.2
        ),
        "demand_level": st.sidebar.selectbox("Demand Level", [0, 1, 2, 3]),
        "is_holiday": st.sidebar.selectbox("Is Holiday", [0, 1]),
        "peak_time_flag": st.sidebar.selectbox("Peak Time Flag", [0, 1]),
        "season": st.sidebar.selectbox("Season", [0, 1, 2, 3]),
        "pickup_location_freq": st.sidebar.number_input(
            "Pickup Location Frequency", 0.0, 1.0, 0.012
        ),
        "drop_location_freq": st.sidebar.number_input(
            "Drop Location Frequency", 0.0, 1.0, 0.010
        ),
        "booking_day": st.sidebar.number_input("Booking Day", 1, 31, 15),
        "booking_month": st.sidebar.number_input("Booking Month", 1, 12, 2),
        "booking_year": st.sidebar.number_input("Booking Year", 2020, 2030, 2026),
        "booking_hour": st.sidebar.slider("Booking Hour", 0, 23, 18),
        "fare_per_km": st.sidebar.number_input("Fare per KM", 0.0, 100.0, 4.0),
        "fare_per_min": st.sidebar.number_input("Fare per Min", 0.0, 100.0, 1.8),
        "surge_demand_interaction": st.sidebar.number_input(
            "Surge Demand Interaction", 0.0, 10.0, 2.8
        ),
        "rush_hour_flag": st.sidebar.selectbox("Rush Hour Flag", [0, 1]),
        "long_distance_flag": st.sidebar.selectbox("Long Distance Flag", [0, 1]),
        "delay_ratio": st.sidebar.number_input("Delay Ratio", 0.0, 1.0, 0.02),
        "surge_high_flag": st.sidebar.selectbox("Surge High Flag", [0, 1]),
    }

    # ======================
    # Prediction Button (Main Page)
    # ======================
    if st.button("Predict Fare 💰"):

        input_df = pd.DataFrame([input_data])
        predicted_fare = pipeline.predict(input_df)[0]

        st.success(f"Predicted Fare: ₹ {round(predicted_fare, 2)}")

        # ======================
        # Bar Chart
        # ======================
        fig, ax = plt.subplots()
        ax.bar(["Predicted Fare"], [predicted_fare])
        ax.set_ylabel("Fare Amount")
        ax.set_title("Fare Prediction Result")

        st.pyplot(fig)
