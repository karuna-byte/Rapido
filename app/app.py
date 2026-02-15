import streamlit as st
from pages.HomePage import home_page
from pages.AnalyticsExplorer import analytics_explorer_page
from pages.RideOutcomePrediction import ride_outcome_prediction_page
from pages.CustomerCancellationPrediction import (
    customer_cancellation_prediction_page,
)
from pages.FarePrediction import fare_prediction_page
from pages.DriverDelayPrediction import driver_delay_prediction_page

st.markdown(
    """
<style>
/* Hide Streamlit multipage navigation */
[data-testid="stSidebarNav"] {
    display: none;
}
</style>
""",
    unsafe_allow_html=True,
)

st.set_page_config(
    page_title="Traffic Violation Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Rapido: Intelligent Mobility Insights")
menu = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Analytics Explorer",
        "Ride Outcome Prediction",
        "Customer Cancellation Prediction",
        "Fare Prediction",
        "Driver Delay Prediction",
    ],
)

if menu == "Home":
    home_page()
elif menu == "Analytics Explorer":
    analytics_explorer_page()
elif menu == "Ride Outcome Prediction":
    ride_outcome_prediction_page()
elif menu == "Fare Prediction":
    fare_prediction_page()
elif menu == "Customer Cancellation Prediction":
    customer_cancellation_prediction_page()
elif menu == "Driver Delay Prediction":
    driver_delay_prediction_page()
