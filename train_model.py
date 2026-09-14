import os
from typing import Any, cast

import joblib # pyright: ignore[reportMissingTypeStubs]
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split # pyright: ignore[reportUnknownVariableType]
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # pyright: ignore[reportUnknownVariableType]


# ============================================================
# PATHS
# ============================================================

# Always use the folder where this script is located.
# This prevents problems caused by running the script from
# a different terminal directory.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "training_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model1.pkl"
)


# ============================================================
# FEATURES
# ============================================================

# These are the same 8 features used by the reference model.

FEATURES = [
    "T",
    "TM",
    "Tm",
    "SLP",
    "H",
    "VV",
    "V",
    "VM"
]

TARGET = "AQI"


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

print(f"\nData file:")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nTraining data was not found:\n{DATA_PATH}\n"
        "Make sure data/training_data.csv exists."
    )

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nThe following required columns are missing:\n"
        f"{missing_columns}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )

print("\nRequired columns found successfully.")


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

print("\nConverting model columns to numeric...")

for column in required_columns:
    series = df[column]
    df[column] = pd.to_numeric(  # pyright: ignore[reportUnknownMemberType]
        series,
        errors="coerce"
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

before_drop = len(df)

df = df.dropna(  # pyright: ignore[reportUnknownMemberType]
    subset=required_columns
).copy()

after_drop = len(df)

print(
    f"Rows removed because of missing/invalid values: "
    f"{before_drop - after_drop}"
)

print(
    f"Rows available for training: {after_drop}"
)


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[FEATURES]
y = df[TARGET]

print("\nFeatures used by the model:")

for feature in FEATURES:
    print(f"  - {feature}")

print(f"\nTarget: {TARGET}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = cast(
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

# This configuration is intentionally compact.
#
# Reference architecture:
# RandomForestRegressor
#
# Compared with the previous 300-tree / unlimited-depth
# configuration, this prevents the model from becoming huge.

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

print("\nModel configuration:")
print("  Algorithm: RandomForestRegressor")
print("  Trees: 100")
print("  Max depth: 15")
print("  Min samples per leaf: 2")
print("  Max features: sqrt")
print("  Random state: 42")
print("  CPU workers: all available")


# ============================================================
# TRAIN
# ============================================================

print("\nTraining model...")
print("Please wait...")

# Cast to Any so static type checkers do not report a partially unknown
# method signature for sklearn estimators while still preserving runtime behavior.
model_any = cast(Any, model)
model_any.fit(
    X_train,
    y_train
)

print("\nTraining completed successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING MODEL")
print("=" * 70)

model_any = cast(Any, model)
y_pred = model_any.predict(X_test)


# ============================================================
# METRICS
# ============================================================

# Convert to NumPy arrays for the metric functions.
# Cast to Any so Pyright does not report a partially unknown pandas Series
# method signature while preserving the runtime behavior.
y_test_array = cast(Any, y_test).to_numpy() if hasattr(y_test, "to_numpy") else y_test
y_pred_array = y_pred

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test_array,
    y_pred_array
) ** 0.5

r2 = r2_score(
    y_test_array,
    y_pred_array
)


print("\nModel performance:")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

model_any = cast(Any, model)
feature_importances = cast(
    list[float],
    list(model_any.feature_importances_)
)

feature_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": feature_importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

for _, row in feature_importance.iterrows():
    print(
        f"  {row['Feature']:>4} : "
        f"{row['Importance']:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

# Remove an existing model first so we don't accidentally
# mistake an old gigantic model for the new one.

if os.path.exists(MODEL_PATH):
    print("\nRemoving previous model...")
    os.remove(MODEL_PATH)


print("\nSaving compressed model...")

# joblib.dump is typed as partially unknown in some environments.
# Accessing it via getattr avoids the member-type warning from type checkers
# while still letting us call it through an Any-typed function reference.

def save_model(value: Any, filename: str, *, compress: int = 0) -> None:
    dump_func = getattr(joblib, "dump")
    dump_func(value, filename, compress=compress)


save_model(
    model,
    MODEL_PATH,
    compress=9,
)

print("\nModel saved successfully.")
print(f"\nSaved to:")
print(MODEL_PATH)


# ============================================================
# MODEL SIZE
# ============================================================

model_size_bytes = os.path.getsize(
    MODEL_PATH
)

model_size_mb = model_size_bytes / (
    1024 * 1024
)

print("\nModel size:")
print(f"  {model_size_mb:.2f} MB")
print(f"  {model_size_bytes:,} bytes")


# ============================================================
# GITHUB SIZE CHECK
# ============================================================

print("\n" + "=" * 70)
print("GITHUB SIZE CHECK")
print("=" * 70)

if model_size_mb < 25:
    print("\n✓ EXCELLENT")
    print("Model is below 25 MB.")
    print("It should be suitable for normal GitHub upload.")

elif model_size_mb < 50:
    print("\n✓ ACCEPTABLE")
    print("Model is below 50 MB.")
    print("However, keeping it below 25 MB is preferable.")

elif model_size_mb < 100:
    print("\n⚠ LARGE")
    print("Model is below GitHub's 100 MB individual-file limit,")
    print("but it is too large for the GitHub browser upload limit.")

else:
    print("\n✗ TOO LARGE")
    print("Model exceeds GitHub's normal individual-file limit.")
    print("Do NOT upload this model yet.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nFinal model:")
print("  Algorithm: RandomForestRegressor")
print("  Trees: 100")
print("  Max depth: 15")
print("  Min samples per leaf: 2")
print("  Max features: sqrt")

print("\nPerformance:")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")

print("\nModel:")
print(f"  Size: {model_size_mb:.2f} MB")
print(f"  Path: {MODEL_PATH}")

print("\n" + "=" * 70)