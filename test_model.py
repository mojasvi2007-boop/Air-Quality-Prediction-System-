import os
from typing import Any, cast

# pyright: reportMissingTypeStubs=false
import joblib
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model1.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("AEROPREDICT MODEL TEST")
print("=" * 60)

print("\nLoading model...")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

model = cast(Any, joblib.load(MODEL_PATH)) # pyright: ignore[reportUnknownMemberType, reportUnnecessaryCast]

print("\nModel loaded successfully.")


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print(f"\nModel type: {type(model).__name__}")

if hasattr(model, "n_estimators"):
    print(f"Trees: {model.n_estimators}")

if hasattr(model, "max_depth"):
    print(f"Maximum depth: {model.max_depth}")

if hasattr(model, "min_samples_leaf"):
    print(f"Minimum samples per leaf: {model.min_samples_leaf}")

if hasattr(model, "max_features"):
    print(f"Max features: {model.max_features}")

if hasattr(model, "n_features_in_"):
    print(f"Number of features: {model.n_features_in_}")

if hasattr(model, "feature_names_in_"):
    print("\nExpected features:")

    for feature in model.feature_names_in_:
        print(f"  - {feature}")


# ============================================================
# TEST INPUT
# ============================================================

# These are the same 8 features used during training.

test_data = pd.DataFrame(
    [[
        30.0,    # T   = Temperature
        34.0,    # TM  = Maximum temperature
        25.0,    # Tm  = Minimum temperature
        1005.0,  # SLP = Sea-level pressure
        70.0,    # H   = Humidity
        8.0,     # VV  = Visibility
        12.0,    # V   = Wind speed
        20.0     # VM  = Maximum wind speed
    ]],
    columns=[
        "T",
        "TM",
        "Tm",
        "SLP",
        "H",
        "VV",
        "V",
        "VM"
    ]
)


# ============================================================
# DISPLAY INPUT
# ============================================================

print("\n" + "=" * 60)
print("TEST INPUT")
print("=" * 60)

print("\n")

for column in test_data.columns:
    print(f"{column:>4} : {test_data[column].iloc[0]}")


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTION")
print("=" * 60)

prediction = model.predict(test_data)

predicted_aqi = float(prediction[0])

print(f"\nPredicted AQI: {predicted_aqi:.2f}")


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderately Polluted"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


category = get_aqi_category(predicted_aqi)

print(f"AQI Category: {category}")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("MODEL TEST COMPLETE")
print("=" * 60)

print("\nThe model loaded and generated a prediction successfully.")