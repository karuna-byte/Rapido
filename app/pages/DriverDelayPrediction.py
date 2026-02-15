import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ======================================================
# PAGE CONFIG
# ======================================================
def driver_delay_prediction_page():
    st.set_page_config(page_title="Driver Delay Prediction", layout="wide")
    st.title("🚦 Driver Delay Prediction System")

    # ======================================================
    # LOAD MODEL
    # ======================================================
    @st.cache_resource
    def load_model():
        return joblib.load("src/models/driver_delay_prediction_model.pkl")

    model = load_model()

    # ======================================================
    # FEATURE ORDER (MUST MATCH TRAINING)
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
    # SIDEBAR INPUTS
    # ======================================================
    st.sidebar.header("Enter Ride & Driver Details")

    input_data = {
        "driver_experience_years": st.sidebar.number_input(
            "Driver Experience (Years)", 0, 40, 4
        ),
        "total_assigned_rides": st.sidebar.number_input(
            "Total Assigned Rides", 0, 10000, 500
        ),
        "accepted_rides": st.sidebar.number_input("Accepted Rides", 0, 10000, 420),
        "avg_driver_rating": st.sidebar.number_input(
            "Average Driver Rating", 0.0, 5.0, 4.6
        ),
        "ride_distance_km": st.sidebar.number_input(
            "Ride Distance (KM)", 0.0, 100.0, 12.0
        ),
        "estimated_ride_time_min": st.sidebar.number_input(
            "Estimated Ride Time (Min)", 0.0, 180.0, 28.0
        ),
        "surge_multiplier": st.sidebar.number_input("Surge Multiplier", 1.0, 5.0, 1.8),
        "hour_of_day": st.sidebar.slider("Hour of Day", 0, 23, 18),
        "is_weekend": st.sidebar.selectbox("Is Weekend", [0, 1]),
        "traffic_level": st.sidebar.selectbox("Traffic Level", [0, 1, 2, 3, 4]),
        "weather_condition": st.sidebar.selectbox(
            "Weather Condition", ["Clear", "Rain", "Fog", "Storm"]
        ),
        "avg_wait_time_min": st.sidebar.number_input(
            "Avg Wait Time (Min)", 0.0, 60.0, 7.0
        ),
        "avg_surge_multiplier": st.sidebar.number_input(
            "Avg Surge Multiplier", 1.0, 5.0, 1.5
        ),
        "demand_level": st.sidebar.selectbox("Demand Level", [1, 2, 3, 4, 5]),
        "peak_time_flag": st.sidebar.selectbox("Peak Time", [0, 1]),
        "is_holiday": st.sidebar.selectbox("Is Holiday", [0, 1]),
        "season": st.sidebar.selectbox(
            "Season", ["Winter", "Summer", "Monsoon", "Autumn"]
        ),
    }

    df_input = pd.DataFrame([input_data])

    # ======================================================
    # DERIVED FEATURES (Same as training)
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

    # Keep correct feature order
    df_input = df_input[features]

    # Convert categorical to numeric codes
    for col in df_input.select_dtypes(include="object").columns:
        df_input[col] = df_input[col].astype("category").cat.codes

    X_input = df_input.values

    # ======================================================
    # MAIN PAGE BUTTON
    # ======================================================
    if st.button("Predict Delay Probability 🚦"):

        probability = model.predict_proba(X_input)[0][1]
        threshold = 0.55
        prediction = 1 if probability >= threshold else 0

        st.subheader("Prediction Result")

        st.write(f"**Delay Probability:** {round(probability, 4)}")

        if prediction == 1:
            st.error("⚠️ Driver Likely To Be Delayed")
        else:
            st.success("✅ Driver Likely On Time")

        # ==================================================
        # BAR CHART
        # ==================================================
        fig, ax = plt.subplots()
        ax.bar(["On Time", "Delayed"], [1 - probability, probability])
        ax.set_ylabel("Probability")
        ax.set_title("Delay Probability Distribution")

        st.pyplot(fig)
