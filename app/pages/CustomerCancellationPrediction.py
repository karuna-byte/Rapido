import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# PAGE CONFIG
# ==========================================
def customer_cancellation_prediction_page():
    st.set_page_config(page_title="Customer Cancellation Prediction", layout="wide")

    st.title("🚖 Customer Ride Cancellation Prediction System")

    # ==========================================
    # LOAD MODEL
    # ==========================================
    @st.cache_resource
    def load_model():
        with open("src/models/customer_cancellation_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model

    model = load_model()
    feature_columns = model.feature_names_in_

    # ==========================================
    # SIDEBAR INPUTS
    # ==========================================
    st.sidebar.header("Enter Ride Details")

    customer_id = st.sidebar.number_input("Customer ID", value=1012)
    driver_id = st.sidebar.number_input("Driver ID", value=455)
    city = st.sidebar.number_input("City Code", value=2)
    pickup_location = st.sidebar.number_input("Pickup Location Code", value=10)
    vehicle_type = st.sidebar.number_input("Vehicle Type", value=1)

    ride_distance_km = st.sidebar.number_input("Ride Distance (km)", value=8.5)
    estimated_ride_time_min = st.sidebar.number_input(
        "Estimated Ride Time (min)", value=20
    )
    booking_value = st.sidebar.number_input("Booking Value", value=240.0)
    surge_multiplier = st.sidebar.number_input("Surge Multiplier", value=1.2)

    hour_of_day = st.sidebar.slider("Hour of Day", 0, 23, 18)
    day_of_week = st.sidebar.slider("Day of Week", 0, 6, 5)
    is_weekend = st.sidebar.selectbox("Is Weekend?", [0, 1])
    booking_month = st.sidebar.slider("Booking Month", 1, 12, 2)
    booking_day = st.sidebar.slider("Booking Day", 1, 31, 13)
    booking_hour = st.sidebar.slider("Booking Hour", 0, 23, 18)

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================

    fare_per_km = booking_value / (ride_distance_km + 1)
    fare_per_min = booking_value / (estimated_ride_time_min + 1)
    long_distance_flag = 1 if ride_distance_km > 15 else 0
    high_surge_flag = 1 if surge_multiplier > 1.5 else 0

    # ==========================================
    # CREATE INPUT DATAFRAME
    # ==========================================

    input_data = {
        "customer_id": customer_id,
        "driver_id": driver_id,
        "city": city,
        "pickup_location": pickup_location,
        "vehicle_type": vehicle_type,
        "ride_distance_km": ride_distance_km,
        "estimated_ride_time_min": estimated_ride_time_min,
        "booking_value": booking_value,
        "surge_multiplier": surge_multiplier,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "booking_month": booking_month,
        "booking_day": booking_day,
        "booking_hour": booking_hour,
        "fare_per_km": fare_per_km,
        "fare_per_min": fare_per_min,
        "long_distance_flag": long_distance_flag,
        "high_surge_flag": high_surge_flag,
    }

    input_df = pd.DataFrame([input_data])

    # Add missing columns if model trained with more
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns exactly as training
    input_df = input_df[feature_columns]

    # ==========================================
    # MAIN PAGE PREDICT BUTTON
    # ==========================================

    if st.button("🚀 Predict Cancellation"):

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        cancel_prob = probability
        complete_prob = 1 - probability

        st.subheader("Prediction Result")

        st.write(f"### Cancellation Probability: {round(cancel_prob * 100, 2)}%")
        st.write(f"### Completion Probability: {round(complete_prob * 100, 2)}%")

        if prediction == 1:
            st.error("🚫 Ride Likely to be Cancelled")
        else:
            st.success("✅ Ride Likely to be Completed")

        # ==========================================
        # BAR CHART
        # ==========================================

        fig, ax = plt.subplots()
        labels = ["Completed", "Cancelled"]
        values = [complete_prob, cancel_prob]

        ax.bar(labels, values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probability Distribution")

        st.pyplot(fig)
