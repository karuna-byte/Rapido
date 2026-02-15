# ======================================================
# 1. IMPORT LIBRARIES
# ======================================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ======================================================
# 2. LOAD DATA
# ======================================================

bookings = pd.read_csv("data/raw/bookings.csv")
customers = pd.read_csv("data/raw/customers.csv")
drivers = pd.read_csv("data/raw/drivers.csv")
location_demand = pd.read_csv("data/raw/location_demand.csv")
time_features = pd.read_csv("data/raw/time_features.csv")


# ======================================================
# 3. CREATE TARGET (BOOKING LEVEL)
# ======================================================

bookings["booking_cancel_flag"] = np.where(
    bookings["booking_status"] == "Cancelled", 1, 0
)


# ======================================================
# 4. MERGE DATASETS (SAFE KEYS)
# ======================================================

df = bookings.merge(customers, on="customer_id", how="left")
df = df.merge(drivers, on="driver_id", how="left", suffixes=("", "_driver"))

# Merge location demand (correct unique keys)
df = df.merge(
    location_demand,
    on=["city", "pickup_location", "hour_of_day", "vehicle_type"],
    how="left",
    suffixes=("", "_demand"),
)

# Merge time features
df = df.merge(
    time_features, on=["hour_of_day", "day_of_week", "is_weekend"], how="left"
)


# ======================================================
# 5. DATETIME FEATURE ENGINEERING
# ======================================================

df["booking_datetime"] = pd.to_datetime(
    df["booking_date"] + " " + df["booking_time"], errors="coerce"
)

df["booking_month"] = df["booking_datetime"].dt.month
df["booking_day"] = df["booking_datetime"].dt.day
df["booking_hour"] = df["booking_datetime"].dt.hour

df = df.drop(["booking_date", "booking_time", "booking_datetime"], axis=1)


# ======================================================
# 6. FEATURE ENGINEERING
# ======================================================

df["fare_per_km"] = df["booking_value"] / (df["ride_distance_km"] + 1)
df["fare_per_min"] = df["booking_value"] / (df["estimated_ride_time_min"] + 1)
df["long_distance_flag"] = np.where(df["ride_distance_km"] > 10, 1, 0)
df["high_surge_flag"] = np.where(df["surge_multiplier"] > 1.5, 1, 0)


# ======================================================
# 7. REMOVE DATA LEAKAGE
# ======================================================

df = df.drop(
    [
        "booking_id",
        "booking_status",  # leakage
        "actual_ride_time_min",  # leakage
        "incomplete_ride_reason",
        "customer_cancel_flag",  # customer-level leakage
    ],
    axis=1,
    errors="ignore",
)


# IMPORTANT:
# Remove customer aggregated leakage columns
leakage_cols = [
    "total_bookings",
    "completed_rides",
    "cancelled_rides",
    "incomplete_rides",
    "cancellation_rate",
    "avg_customer_rating",
]

df = df.drop(leakage_cols, axis=1, errors="ignore")


# ======================================================
# 8. HANDLE MISSING VALUES
# ======================================================

df = df.fillna(0)


# ======================================================
# 9. ENCODE CATEGORICAL COLUMNS
# ======================================================

categorical_cols = df.select_dtypes(include=["object"]).columns

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))


# ======================================================
# 10. FINAL SAFETY CHECK (NO DATETIME)
# ======================================================

df = df.select_dtypes(exclude=["datetime64[ns]"])


# ======================================================
# 11. DEFINE FEATURES & TARGET
# ======================================================

target = "booking_cancel_flag"

X = df.drop(target, axis=1)
y = df[target]


# ======================================================
# 12. TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ======================================================
# 13. TRAIN MODEL
# ======================================================

model = RandomForestClassifier(
    n_estimators=300, max_depth=15, class_weight="balanced", random_state=42
)

model.fit(X_train, y_train)


# ======================================================
# 14. PREDICTIONS
# ======================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ======================================================
# 15. EVALUATION
# ======================================================

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n====================================")
print("Booking Cancellation Prediction Model")
print("====================================")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("F1 Score :", round(f1, 4))
print("AUC Score:", round(auc, 4))
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ======================================================
# 16. FEATURE IMPORTANCE
# ======================================================

importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
).sort_values(by="Importance", ascending=False)

print("\nTop 15 Important Features:")
print(importance.head(15))

with open("customer_cancellation_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

# Accuracy : 87.29 %
# F1 Score : 0.7815
# AUC Score: 0.978

# Confusion Matrix:
#  [[673164 126902]
#  [  5684 237162]]
