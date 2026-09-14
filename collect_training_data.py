import os
import time
from typing import Any, cast

import requests
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Historical Forecast API
# This endpoint provides archived forecast-model data and
# includes visibility, which the normal Historical Weather API
# did not provide for our previous request.
WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Open-Meteo Air Quality API
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# ------------------------------------------------------------
# Date range
# ------------------------------------------------------------
# One year of data.
# Historical Forecast API has coverage from around 2022 onward.
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
# Always save data relative to this Python file,
# regardless of the terminal's current directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "training_data.csv")

# ------------------------------------------------------------
# Indian cities
# ------------------------------------------------------------

LOCATIONS: dict[str, tuple[float, float]] = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Chandigarh": (30.7333, 76.7794),
    "Bhopal": (23.2599, 77.4126),
    "Patna": (25.5941, 85.1376),
    "Bhubaneswar": (20.2961, 85.8245),
}


# ============================================================
# API SETTINGS
# ============================================================

WEATHER_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "visibility",
    "wind_speed_10m",
]

AQI_PARAMS = [
    "us_aqi",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def request_json(
    url: str,
    params: dict[str, str | int | float],
    description: str,
) -> dict[str, object] | None:
    """
    Send a GET request and return JSON.
    Includes basic retry handling.
    """

    for attempt in range(3):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            if data.get("error"):
                raise RuntimeError(
                    f"{description} API error: "
                    f"{data.get('reason', 'Unknown error')}"
                )

            return data

        except Exception as error:

            print(
                f"  Request failed for {description} "
                f"(attempt {attempt + 1}/3): {error}"
            )

            if attempt < 2:
                time.sleep(5)

    return None


# ============================================================
# WEATHER DATA
# ============================================================

def get_weather_data(city: str, latitude: float, longitude: float):

    print(f"\n  Downloading weather data for {city}...")

    params: dict[str, str | int | float] = {
        "latitude": latitude,
        "longitude": longitude,

        "start_date": START_DATE,
        "end_date": END_DATE,

        "hourly": ",".join(WEATHER_PARAMS),

        "timezone": "UTC",

        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    data = request_json(
        WEATHER_URL,
        params,
        f"weather - {city}"
    )

    if data is None:
        return None

    if "hourly" not in data:
        print(f"  ERROR: No hourly weather data for {city}")
        return None

    hourly_value = data["hourly"]
    if not isinstance(hourly_value, dict):
        print(f"  ERROR: Invalid hourly weather payload for {city}")
        return None

    hourly = cast(dict[str, Any], hourly_value)

    if "time" not in hourly:
        print(f"  ERROR: No timestamps returned for {city}")
        return None

    df = pd.DataFrame({
        "DateTime": hourly["time"],
        "T": hourly.get("temperature_2m"),
        "H": hourly.get("relative_humidity_2m"),
        "SLP": hourly.get("pressure_msl"),
        "VV": hourly.get("visibility"),
        "V": hourly.get("wind_speed_10m"),
    })

    # --------------------------------------------------------
    # Convert timestamp to UTC hourly timestamp
    # --------------------------------------------------------

    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        utc=True,
        errors="coerce"
    )

    df["DateTime"] = (
        df["DateTime"]
        .dt.floor("h")
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    weather_columns = [
        "T",
        "H",
        "SLP",
        "VV",
        "V",
    ]

    for column in weather_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Convert visibility from metres to kilometres
    #
    # Open-Meteo visibility is returned in metres.
    # The reference model uses VV in a visibility-like unit,
    # and the frontend uses kilometres.
    # --------------------------------------------------------

    df["VV"] = df["VV"] / 1000.0

    # --------------------------------------------------------
    # Calculate daily features required by the reference model
    #
    # TM = daily maximum temperature
    # Tm = daily minimum temperature
    # VM = daily maximum wind speed
    # --------------------------------------------------------

    df["Date"] = df["DateTime"].dt.date

    daily = (
        df.groupby("Date")
        .agg(
            TM=("T", "max"),
            Tm=("T", "min"),
            VM=("V", "max"),
        )
        .reset_index()
    )

    # Merge daily values back into hourly dataframe

    df = df.merge(
        daily,
        on="Date",
        how="left"
    )

    df.drop(
        columns=["Date"],
        inplace=True
    )

    print(f"  Weather timestamps: {len(df):,}")

    # Visibility diagnostic

    visibility_missing = df["VV"].isna().sum()

    print(
        f"  Missing visibility: "
        f"{visibility_missing:,} "
        f"({visibility_missing / len(df) * 100:.2f}%)"
    )

    return df


# ============================================================
# AIR QUALITY DATA
# ============================================================

def get_air_quality_data(city: str, latitude: float, longitude: float):

    print(f"  Downloading AQI data for {city}...")

    params: dict[str, str | int | float] = {
        "latitude": latitude,
        "longitude": longitude,

        "start_date": START_DATE,
        "end_date": END_DATE,

        "hourly": ",".join(AQI_PARAMS),

        "timezone": "UTC",
    }

    data = request_json(
        AIR_QUALITY_URL,
        params,
        f"air quality - {city}"
    )

    if data is None:
        return None

    if "hourly" not in data:
        print(f"  ERROR: No AQI data for {city}")
        return None

    hourly = cast(dict[str, Any], data["hourly"])

    if "time" not in hourly:
        print(f"  ERROR: No AQI timestamps for {city}")
        return None

    df = pd.DataFrame({
        "DateTime": hourly["time"],
        "AQI": hourly.get("us_aqi"),
    })

    # --------------------------------------------------------
    # Timestamp normalization
    # --------------------------------------------------------

    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        utc=True,
        errors="coerce"
    )

    df["DateTime"] = (
        df["DateTime"]
        .dt.floor("h")
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    df["AQI"] = pd.to_numeric(
        df["AQI"],
        errors="coerce"
    )

    print(f"  AQI timestamps: {len(df):,}")

    return df


# ============================================================
# COMBINE WEATHER + AQI
# ============================================================

def combine_city_data(city: str, latitude: float, longitude: float):

    print("\n" + "=" * 60)
    print(f"PROCESSING: {city}")
    print("=" * 60)

    weather_df = get_weather_data(
        city,
        latitude,
        longitude
    )

    if weather_df is None:
        print(f"  SKIPPING {city}: weather data unavailable.")
        return None

    aqi_df = get_air_quality_data(
        city,
        latitude,
        longitude
    )

    if aqi_df is None:
        print(f"  SKIPPING {city}: AQI data unavailable.")
        return None

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    weather_df = (
        weather_df
        .drop_duplicates(subset=["DateTime"])
        .sort_values("DateTime")
    )

    aqi_df = (
        aqi_df
        .drop_duplicates(subset=["DateTime"])
        .sort_values("DateTime")
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    df = pd.merge(
        weather_df,
        aqi_df,
        on="DateTime",
        how="inner"
    )

    # --------------------------------------------------------
    # Add city
    # --------------------------------------------------------

    df["City"] = city

    # --------------------------------------------------------
    # Expected model columns
    # --------------------------------------------------------

    feature_columns = [
        "T",
        "TM",
        "Tm",
        "SLP",
        "H",
        "VV",
        "V",
        "VM",
    ]

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\n  After merge:")

    print(f"  Weather rows: {len(weather_df):,}")
    print(f"  AQI rows:     {len(aqi_df):,}")
    print(f"  Merged rows:  {len(df):,}")

    print("\n  Missing values:")

    for column in feature_columns + ["AQI"]:

        missing = df[column].isna().sum()

        percentage = (
            missing / len(df) * 100
            if len(df) > 0
            else 0
        )

        print(
            f"    {column:<4}: "
            f"{missing:,} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Clean numeric columns
    # --------------------------------------------------------

    for column in feature_columns + ["AQI"]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Handle weather missing values
    #
    # We may interpolate small gaps in weather variables.
    # We DO NOT interpolate AQI because that would manufacture
    # target values.
    # --------------------------------------------------------

    weather_feature_columns = [
        "T",
        "TM",
        "Tm",
        "SLP",
        "H",
        "VV",
        "V",
        "VM",
    ]

    df[weather_feature_columns] = (
        df[weather_feature_columns]
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )

    # --------------------------------------------------------
    # Drop rows where AQI or model features are still missing
    # --------------------------------------------------------

    before_cleaning = len(df)

    df = df.dropna(
        subset=feature_columns + ["AQI"]
    )

    removed = before_cleaning - len(df)

    print(
        f"\n  Rows removed after cleaning: "
        f"{removed:,}"
    )

    print(
        f"  Usable rows for {city}: "
        f"{len(df):,}"
    )

    if len(df) == 0:

        print(
            f"  ERROR: No usable rows remain for {city}."
        )

        return None

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    df = df[
        [
            "T",
            "TM",
            "Tm",
            "SLP",
            "H",
            "VV",
            "V",
            "VM",
            "AQI",
            "City",
            "DateTime",
        ]
    ]

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AEROPREDICT - TRAINING DATA COLLECTION")
    print("=" * 70)

    print(f"\nDate range:")
    print(f"  Start: {START_DATE}")
    print(f"  End:   {END_DATE}")

    print(f"\nWeather API:")
    print(f"  {WEATHER_URL}")

    print(f"\nCities:")
    print(f"  {len(LOCATIONS)}")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process cities
    # --------------------------------------------------------

    all_city_data: list[pd.DataFrame] = []

    for city, coordinates in LOCATIONS.items():

        latitude, longitude = coordinates

        city_df = combine_city_data(
            city,
            latitude,
            longitude
        )

        if city_df is not None:

            all_city_data.append(
                city_df
            )

        # Small delay to avoid hammering APIs
        time.sleep(2)

    # --------------------------------------------------------
    # Check whether anything was collected
    # --------------------------------------------------------

    if not all_city_data:

        print("\n" + "=" * 70)
        print("ERROR: NO DATA WAS COLLECTED")
        print("=" * 70)

        return

    # --------------------------------------------------------
    # Combine all cities
    # --------------------------------------------------------

    final_df = pd.concat(
        all_city_data,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    final_df = final_df.sort_values(
        [
            "City",
            "DateTime",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Final numeric cleanup
    # --------------------------------------------------------

    numeric_columns = [
        "T",
        "TM",
        "Tm",
        "SLP",
        "H",
        "VV",
        "V",
        "VM",
        "AQI",
    ]

    for column in numeric_columns:

        final_df[column] = pd.to_numeric(
            final_df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DATASET VALIDATION")
    print("=" * 70)

    print(
        f"\nTotal rows: "
        f"{len(final_df):,}"
    )

    print(
        f"Total cities: "
        f"{final_df['City'].nunique()}"
    )

    print("\nRows per city:")

    print(
        final_df["City"]
        .value_counts()
        .sort_index()
    )

    print("\nMissing values:")

    print(
        final_df.isna()
        .sum()
    )

    print("\nFeature statistics:")

    print(
        final_df[numeric_columns]
        .describe()
        .round(2)
    )

    # --------------------------------------------------------
    # Remove remaining invalid rows
    # --------------------------------------------------------

    final_df = final_df.dropna(
        subset=numeric_columns
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final confirmation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved to:"
        f"\n  {os.path.abspath(OUTPUT_FILE)}"
    )

    print(
        f"\nFinal rows:"
        f"\n  {len(final_df):,}"
    )

    print(
        "\nColumns:"
    )

    print(
        list(final_df.columns)
    )

    print("\nFirst 5 rows:")

    print(
        final_df.head().to_string(
            index=False
        )
    )

    print("\nDone.")


if __name__ == "__main__":
    main()