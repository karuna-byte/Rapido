import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data


def analytics_explorer_page():
    st.set_page_config(page_title="Rapido EDA Dashboard", layout="wide")

    st.title("📊 Rapido Booking Data - Exploratory Data Analysis")
    st.markdown("---")

    # =========================
    # LOAD DATA
    # =========================
    # @st.cache_data
    # def load_data():
    #     bookings = pd.read_csv("data/raw/bookings.csv")
    #     customers = pd.read_csv("data/raw/customers.csv")
    #     drivers = pd.read_csv("data/raw/drivers.csv")
    #     location_demand = pd.read_csv("data/raw/location_demand.csv")
    #     time_features = pd.read_csv("data/raw/time_features.csv")
    #     return bookings, customers, drivers, location_demand, time_features

    bookings, customers, drivers, location_demand, time_features = load_data()

    # =========================
    # DATA PREP
    # =========================
    bookings["is_cancelled"] = bookings["booking_status"].apply(
        lambda x: (
            1 if x in ["Cancelled", "Driver Cancelled", "Customer Cancelled"] else 0
        )
    )

    # =========================
    # 1️⃣ Ride Volume Analysis
    # =========================
    st.header("🚗 Ride Volume Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ride Volume by Hour")
        hour_data = bookings.groupby("hour_of_day").size()
        fig, ax = plt.subplots()
        hour_data.plot(kind="bar", ax=ax)
        ax.set_ylabel("Total Rides")
        st.pyplot(fig)

    with col2:
        st.subheader("Ride Volume by Day of Week")
        day_data = bookings.groupby("day_of_week").size()
        fig, ax = plt.subplots()
        day_data.plot(kind="bar", ax=ax)
        ax.set_ylabel("Total Rides")
        st.pyplot(fig)

    st.markdown("---")

    # =========================
    # 2️⃣ Cancellation Heatmap
    # =========================
    st.header("🔥 Cancellation Rate Across Cities")

    cancel_city = bookings.groupby("city")["is_cancelled"].mean().reset_index()

    fig, ax = plt.subplots()
    sns.heatmap(cancel_city.set_index("city"), annot=True, cmap="Reds", ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    # =========================
    # 3️⃣ Distance vs Fare
    # =========================
    st.header("📏 Distance vs Booking Value Correlation")

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=bookings.sample(5000), x="ride_distance_km", y="booking_value", ax=ax
    )
    st.pyplot(fig)

    correlation = bookings[["ride_distance_km", "booking_value"]].corr().iloc[0, 1]
    st.write(f"Correlation Coefficient: **{round(correlation, 3)}**")

    st.markdown("---")

    # =========================
    # 4️⃣ Rating Distribution
    # =========================
    st.header("⭐ Rating Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Ratings")
        fig, ax = plt.subplots()
        customers["avg_customer_rating"].hist(bins=20, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Driver Ratings")
        fig, ax = plt.subplots()
        drivers["avg_driver_rating"].hist(bins=20, ax=ax)
        st.pyplot(fig)

    st.markdown("---")

    # =========================
    # 5️⃣ Customer vs Driver Behaviour
    # =========================
    st.header("👥 Customer vs Driver Behaviour Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Cancellation Rate")
        fig, ax = plt.subplots()
        customers["cancellation_rate"].hist(bins=20, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Driver Delay Rate")
        fig, ax = plt.subplots()
        drivers["delay_rate"].hist(bins=20, ax=ax)
        st.pyplot(fig)

    st.markdown("---")

    # =========================
    # 6️⃣ Traffic / Weather vs Cancellation
    # =========================
    st.header("🌧 Traffic & Weather Impact on Cancellation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Traffic Level vs Cancellation")
        traffic_cancel = bookings.groupby("traffic_level")["is_cancelled"].mean()
        fig, ax = plt.subplots()
        traffic_cancel.plot(kind="bar", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Weather Condition vs Cancellation")
        weather_cancel = bookings.groupby("weather_condition")["is_cancelled"].mean()
        fig, ax = plt.subplots()
        weather_cancel.plot(kind="bar", ax=ax)
        st.pyplot(fig)

    st.markdown("---")
