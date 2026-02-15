import streamlit as st
import pickle
import pandas as pd


# ======================================================
# PAGE CONFIG
# ======================================================
def ride_outcome_prediction_page():
    st.set_page_config(
        page_title="Ride Outcome Prediction", page_icon="🚖", layout="wide"
    )

    st.title("🚖 Ride Outcome Prediction System")
    st.markdown(
        "Predict whether a ride will be **Completed, Cancelled, or Incomplete**"
    )

    # ======================================================
    # LOAD MODEL & FEATURES
    # ======================================================

    @st.cache_resource
    def load_model():
        with open("src/models/ride_outcome_prediction_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("src/models/ride_outcome_features.pkl", "rb") as f:
            features = pickle.load(f)
        return model, features

    model, FEATURES_STRICT = load_model()

    # ======================================================
    # USER INPUT SECTION
    # ======================================================

    st.sidebar.header("Enter Ride Details")

    day_of_week = st.sidebar.selectbox("Day of Week", list(range(7)))
    is_weekend = st.sidebar.selectbox("Is Weekend", [0, 1])
    hour_of_day = st.sidebar.slider("Hour of Day", 0, 23, 18)

    booking_day = st.sidebar.slider("Booking Day", 1, 31, 15)
    booking_month = st.sidebar.slider("Booking Month", 1, 12, 2)
    booking_year = st.sidebar.number_input("Booking Year", 2024, 2030, 2026)

    is_holiday = st.sidebar.selectbox("Is Holiday", [0, 1])
    peak_time_flag = st.sidebar.selectbox("Peak Time Flag", [0, 1])
    season = st.sidebar.selectbox("Season (Encoded)", [0, 1, 2, 3])

    ride_distance_km = st.sidebar.number_input("Ride Distance (KM)", 0.0, 100.0, 12.5)
    estimated_ride_time_min = st.sidebar.number_input(
        "Estimated Time (Min)", 1.0, 180.0, 28.0
    )

    traffic_level = st.sidebar.selectbox("Traffic Level", [0, 1, 2, 3])
    weather_condition = st.sidebar.selectbox(
        "Weather Condition (Encoded)", [0, 1, 2, 3]
    )

    base_fare = st.sidebar.number_input("Base Fare", 0.0, 10000.0, 50.0)
    surge_multiplier = st.sidebar.number_input("Surge Multiplier", 1.0, 5.0, 1.4)
    booking_value = st.sidebar.number_input("Booking Value", 0.0, 50000.0, 210.0)

    # ======================================================
    # DERIVED FEATURES
    # ======================================================

    fare_per_km = base_fare / (ride_distance_km + 1)
    fare_per_min = base_fare / (estimated_ride_time_min + 1)

    rush_hour_flag = 1 if hour_of_day in [8, 9, 10, 17, 18, 19] else 0
    long_distance_flag = 1 if ride_distance_km > 15 else 0
    surge_high_flag = 1 if surge_multiplier > 1.5 else 0

    # ======================================================
    # CREATE INPUT DATAFRAME
    # ======================================================

    input_data = {
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "hour_of_day": hour_of_day,
        "booking_day": booking_day,
        "booking_month": booking_month,
        "booking_year": booking_year,
        "booking_hour": hour_of_day,
        "is_holiday": is_holiday,
        "peak_time_flag": peak_time_flag,
        "season": season,
        "ride_distance_km": ride_distance_km,
        "estimated_ride_time_min": estimated_ride_time_min,
        "traffic_level": traffic_level,
        "weather_condition": weather_condition,
        "base_fare": base_fare,
        "surge_multiplier": surge_multiplier,
        "booking_value": booking_value,
        "fare_per_km": fare_per_km,
        "fare_per_min": fare_per_min,
        "rush_hour_flag": rush_hour_flag,
        "long_distance_flag": long_distance_flag,
        "surge_high_flag": surge_high_flag,
    }

    input_df = pd.DataFrame([input_data])

    # Enforce strict order
    input_df = input_df[FEATURES_STRICT]

    # ======================================================
    # PREDICTION BUTTON
    # ======================================================

    if st.button("Predict Ride Outcome"):

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        status_map = {0: "❌ Cancelled", 1: "✅ Completed", 2: "⚠️ Incomplete"}

        st.subheader("Prediction Result")
        st.success(f"Predicted Outcome: {status_map[prediction]}")

        st.subheader("Prediction Probabilities")

        prob_df = pd.DataFrame(
            {"Outcome": ["Cancelled", "Completed", "Incomplete"], "Probability": proba}
        )

        st.bar_chart(prob_df.set_index("Outcome"))
