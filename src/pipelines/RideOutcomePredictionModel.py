import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier

# =====================================================
# LOAD DATA
# =====================================================
bookings = pd.read_csv("data/raw/bookings.csv")
customers = pd.read_csv("data/raw/customers.csv")
drivers = pd.read_csv("data/raw/drivers.csv")
location_demand = pd.read_csv("data/raw/location_demand.csv")
time_features = pd.read_csv("data/raw/time_features.csv")

# =====================================================
# MERGE DATA (LEFT JOIN – NO ROW LOSS)
# =====================================================
df = bookings.merge(customers, on="customer_id", how="left")
df = df.merge(drivers, on="driver_id", how="left", suffixes=("", "_driver"))

df = df.merge(
    location_demand,
    on=["city", "pickup_location", "hour_of_day", "vehicle_type"],
    how="left",
)

df = df.merge(
    time_features, on=["hour_of_day", "day_of_week", "is_weekend"], how="left"
)

# =====================================================
# TARGET
# =====================================================
y = df["customer_cancel_flag"]  # 1 = Cancelled, 0 = Completed

# =====================================================
# FEATURE SELECTION (NO LEAKAGE)
# =====================================================
NUM_FEATURES = [
    "ride_distance_km",
    "estimated_ride_time_min",
    "base_fare",
    "surge_multiplier",
    "booking_value",
    "customer_age",
    "customer_signup_days_ago",
    "total_bookings",
    "completed_rides",
    "cancelled_rides",
    "acceptance_rate",
    "delay_rate",
    "avg_driver_rating",
    "avg_pickup_delay_min",
    "avg_wait_time_min",
    "avg_surge_multiplier",
    "cancellation_rate",
    "avg_customer_rating",
    "driver_age",
    "driver_experience_years",
    "delay_count",
    "is_weekend",
    "hour_of_day",
    "is_holiday",
    "peak_time_flag",
]

CAT_FEATURES = [
    "city",
    "pickup_location",
    "drop_location",
    "vehicle_type",
    "traffic_level",
    "weather_condition",
    "day_of_week",
    "season",
    "demand_level",
]

NUM_FEATURES = [c for c in NUM_FEATURES if c in df.columns]
CAT_FEATURES = [c for c in CAT_FEATURES if c in df.columns]

X = df[NUM_FEATURES + CAT_FEATURES].copy()

# =====================================================
# HANDLE MISSING VALUES
# =====================================================
X[NUM_FEATURES] = X[NUM_FEATURES].fillna(0)
X[CAT_FEATURES] = X[CAT_FEATURES].fillna("unknown")

# =====================================================
# ENCODE CATEGORICAL FEATURES
# =====================================================
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

X[CAT_FEATURES] = encoder.fit_transform(X[CAT_FEATURES])

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =====================================================
# MODEL (STABLE 85–90% ACCURACY)
# =====================================================
model = HistGradientBoostingClassifier(
    max_depth=10,
    learning_rate=0.08,
    max_iter=300,
    min_samples_leaf=200,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================
y_prob = model.predict_proba(X_test)[:, 1]
THRESHOLD = 0.55
y_pred = (y_prob >= THRESHOLD).astype(int)

print("\nModel Performance")
print("----------------")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =====================================================
# SAVE MODEL
# =====================================================
# with open("ride_outcome_prediction_model.pkl", "wb") as f:
#     pickle.dump(
#         {
#             "model": model,
#             "encoder": encoder,
#             "features": NUM_FEATURES + CAT_FEATURES,
#             "threshold": THRESHOLD,
#         },
#         f,
#     )

# print("\n✅ Ride outcome prediction Model saved successfully")
