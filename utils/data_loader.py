import pandas as pd
import streamlit as st


# =========================
# LOAD DATA
# =========================
@st.cache_data(show_spinner="Loading data...")
def load_data():
    bookings = pd.read_csv("data/raw/bookings.csv")
    customers = pd.read_csv("data/raw/customers.csv")
    drivers = pd.read_csv("data/raw/drivers.csv")
    location_demand = pd.read_csv("data/raw/location_demand.csv")
    time_features = pd.read_csv("data/raw/time_features.csv")
    return bookings, customers, drivers, location_demand, time_features
