import streamlit as st


# ================================
# PAGE CONFIG
# ================================
def home_page():
    st.set_page_config(
        page_title="Rapido ML Decision System", page_icon="🚀", layout="wide"
    )

    # ================================
    # HEADER
    # ================================
    st.title("🚀 Rapido Unified ML Decision System")
    st.markdown("---")

    st.markdown(
        """
    ### Transforming Ride Operations with Machine Learning

    Rapido operates a large-scale ride-hailing platform where millions of bookings are created daily 
    across multiple cities, vehicle types, and demand conditions.

    This system leverages advanced Machine Learning models to proactively predict ride outcomes, 
    optimize pricing, and enhance customer experience.
    """
    )

    st.markdown("---")

    # ================================
    # OBJECTIVES SECTION
    # ================================
    st.header("🎯 Project Objectives")

    st.markdown(
        """
    ✔ **Predict Ride Outcomes Before Trip Start**  
    ✔ **Estimate Accurate Fares Dynamically**  
    ✔ **Identify High-Risk Customers and Drivers**  
    ✔ **Enable Data-Driven Operational Decisions**
    """
    )

    st.markdown("---")

    # ================================
    # BUSINESS PROBLEM SECTION
    # ================================
    st.header("📊 Business Challenges Addressed")

    st.markdown(
        """
    - High Ride Cancellation Rates  
    - Inaccurate Fare Estimations  
    - Inefficient Driver Allocation  
    - Peak Demand Imbalance  
    - Poor Customer Experience  
    """
    )

    st.markdown("---")

    # ================================
    # SYSTEM CAPABILITIES
    # ================================
    st.header("🧠 ML Capabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Predictive Models")
        st.write(
            """
        - Ride Cancellation Prediction  
        - Driver Delay Prediction  
        - Booking Success Probability  
        """
        )

    with col2:
        st.subheader("💰 Intelligent Pricing")
        st.write(
            """
        - Dynamic Fare Estimation  
        - Surge Demand Detection  
        - Revenue Optimization  
        """
        )

    st.markdown("---")
