import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, cast

import joblib  # pyright: ignore[reportMissingTypeStubs]
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# AEROPREDICT
# AI-BASED AIR QUALITY PREDICTION SYSTEM
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model1.pkl"
)


# ============================================================
# API URLS
# ============================================================

OPEN_METEO_WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

OPEN_METEO_AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/reverse"
)


# ============================================================
# MODEL FEATURES
# ============================================================

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


# ============================================================
# HTTP HEADERS
# ============================================================

REQUEST_HEADERS = {
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "User-Agent": "AeroPredict/1.0"
}


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AeroPredict API",
    description=(
        "AI-based air quality prediction using "
        "Random Forest and live atmospheric data."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

# Required because the frontend may be running through
# VS Code Live Server on a different port.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_model() -> Any:

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print("\n" + "=" * 70)
    print("LOADING AEROPREDICT MODEL")
    print("=" * 70)

    print(f"\nModel path:")
    print(MODEL_PATH)

    # The model was saved using joblib with compression.
    # pyright can flag joblib.load() as a partially unknown member type,
    # so cast the result to Any before use.
    model = cast(Any, joblib.load(MODEL_PATH)) # pyright: ignore[reportUnknownMemberType, reportUnnecessaryCast]

    print("\nModel loaded successfully.")

    if hasattr(model, "n_estimators"):
        n_estimators = cast(Any, getattr(model, "n_estimators", None))
        print(f"Trees: {n_estimators}")

    if hasattr(model, "max_depth"):
        max_depth = cast(Any, getattr(model, "max_depth", None))
        print(f"Maximum depth: {max_depth}")

    if hasattr(model, "min_samples_leaf"):
        min_samples_leaf = cast(Any, getattr(model, "min_samples_leaf", None))
        print(f"Minimum samples per leaf: {min_samples_leaf}")

    if hasattr(model, "n_features_in_"):
        n_features_in = cast(Any, getattr(model, "n_features_in_", None))
        print(f"Number of features: {n_features_in}")

    if hasattr(model, "feature_names_in_"):
        feature_names = cast(Any, getattr(model, "feature_names_in_", None))
        print("Expected features: " + ", ".join(cast(list[str], feature_names)))

    return model


# ============================================================
# NUMBER VALIDATION
# ============================================================

def require_float(
    value: Any,
    field_name: str
) -> float:

    if value is None:
        raise ValueError(
            f"Required value unavailable: {field_name}"
        )

    try:
        return float(value)

    except (TypeError, ValueError) as error:

        raise ValueError(
            f"Invalid value for {field_name}: {value}"
        ) from error


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(
    aqi: float
) -> dict[str, str]:

    if aqi <= 50:

        return {
            "category": "Good",
            "description": (
                "Air quality is considered satisfactory "
                "with little or no health risk."
            )
        }

    if aqi <= 100:

        return {
            "category": "Moderate",
            "description": (
                "Air quality is acceptable. "
                "Some unusually sensitive individuals "
                "may experience minor effects."
            )
        }

    if aqi <= 150:

        return {
            "category": (
                "Unhealthy for Sensitive Groups"
            ),
            "description": (
                "Sensitive groups may experience "
                "health effects. Consider reducing "
                "prolonged outdoor exposure."
            )
        }

    if aqi <= 200:

        return {
            "category": "Unhealthy",
            "description": (
                "Everyone may begin to experience "
                "health effects. Sensitive groups "
                "should reduce prolonged outdoor exposure."
            )
        }

    if aqi <= 300:

        return {
            "category": "Very Unhealthy",
            "description": (
                "Health alert. Everyone may experience "
                "more serious health effects."
            )
        }

    return {
        "category": "Hazardous",
        "description": (
            "Serious health risk. Avoid outdoor exposure "
            "where possible."
        )
    }


# ============================================================
# HEALTH GUIDANCE
# ============================================================

def get_health_guidance(
    aqi: float
) -> dict[str, str]:

    if aqi <= 50:

        return {
            "title": "Air Quality is Good",
            "message": (
                "Outdoor activity is generally suitable "
                "for most people."
            )
        }

    if aqi <= 100:

        return {
            "title": "Air Quality is Moderate",
            "message": (
                "Most people can continue normal outdoor "
                "activities. Sensitive individuals should "
                "monitor symptoms."
            )
        }

    if aqi <= 150:

        return {
            "title": (
                "Air Quality is Unhealthy "
                "for Sensitive Groups"
            ),
            "message": (
                "Children, older adults and sensitive "
                "individuals should consider reducing "
                "prolonged outdoor exertion."
            )
        }

    if aqi <= 200:

        return {
            "title": "Air Quality is Unhealthy",
            "message": (
                "Sensitive groups should reduce prolonged "
                "outdoor exposure. Others should reduce "
                "heavy or extended outdoor exertion."
            )
        }

    if aqi <= 300:

        return {
            "title": "Air Quality is Very Unhealthy",
            "message": (
                "Everyone should reduce outdoor exposure "
                "and avoid prolonged physical activity."
            )
        }

    return {
        "title": "Air Quality is Hazardous",
        "message": (
            "Avoid outdoor exposure where possible "
            "and follow local health advisories."
        )
    }


# ============================================================
# GET CURRENT AIR QUALITY
# ============================================================

def get_current_air_quality(
    latitude: float,
    longitude: float
) -> dict[str, Any]:

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,

        "current": (
            "us_aqi,"
            "pm2_5,"
            "pm10,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone"
        ),

        "timezone": "auto"
    }

    try:

        response = requests.get(
            OPEN_METEO_AIR_QUALITY_URL,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=20
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise ValueError(
            "Unable to retrieve air quality data: "
            f"{error}"
        ) from error

    data = response.json()

    current = data.get(
        "current",
        {}
    )

    return {
        "aqi": require_float(
            current.get("us_aqi"),
            "us_aqi"
        ),

        "pm2_5": require_float(
            current.get("pm2_5"),
            "pm2_5"
        ),

        "pm10": require_float(
            current.get("pm10"),
            "pm10"
        ),

        "carbon_monoxide": require_float(
            current.get("carbon_monoxide"),
            "carbon_monoxide"
        ),

        "nitrogen_dioxide": require_float(
            current.get("nitrogen_dioxide"),
            "nitrogen_dioxide"
        ),

        "sulphur_dioxide": require_float(
            current.get("sulphur_dioxide"),
            "sulphur_dioxide"
        ),

        "ozone": require_float(
            current.get("ozone"),
            "ozone"
        ),

        "timestamp": current.get("time"),

        "timezone": data.get(
            "timezone",
            "Unknown"
        ),

        "data_latitude": data.get(
            "latitude"
        ),

        "data_longitude": data.get(
            "longitude"
        )
    }


# ============================================================
# GET WEATHER FORECAST
# ============================================================

def get_weather_forecast(
    latitude: float,
    longitude: float
) -> dict[str, Any]:

    params: dict[str, Any] = {

        "latitude": latitude,

        "longitude": longitude,

        # Hourly features required by the model
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "pressure_msl,"
            "visibility,"
            "wind_speed_10m"
        ),

        # Daily features required by the model
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "wind_speed_10m_max"
        ),

        "forecast_days": 4,

        "timezone": "auto",

        "wind_speed_unit": "kmh"
    }

    try:

        response = requests.get(
            OPEN_METEO_WEATHER_URL,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=20
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise ValueError(
            "Unable to retrieve weather forecast: "
            f"{error}"
        ) from error

    data = response.json()

    if "hourly" not in data:

        raise ValueError(
            "Weather API returned no hourly data."
        )

    if "daily" not in data:

        raise ValueError(
            "Weather API returned no daily data."
        )

    return data


# ============================================================
# BUILD HOURLY WEATHER DATAFRAME
# ============================================================

def build_weather_dataframe(
    weather_data: dict[str, Any]
) -> pd.DataFrame:

    hourly = weather_data["hourly"]

    daily = weather_data["daily"]

    hourly_df = pd.DataFrame({

        "time": hourly.get("time", []),

        "T": hourly.get(
            "temperature_2m",
            []
        ),

        "H": hourly.get(
            "relative_humidity_2m",
            []
        ),

        "SLP": hourly.get(
            "pressure_msl",
            []
        ),

        "VV": [
            value / 1000
            if value is not None
            else None
            for value in hourly.get(
                "visibility",
                []
            )
        ],

        "V": hourly.get(
            "wind_speed_10m",
            []
        )
    })

    daily_df = pd.DataFrame({

        "date": daily.get(
            "time",
            []
        ),

        "TM": daily.get(
            "temperature_2m_max",
            []
        ),

        "Tm": daily.get(
            "temperature_2m_min",
            []
        ),

        "VM": daily.get(
            "wind_speed_10m_max",
            []
        )
    })

    if hourly_df.empty:

        raise ValueError(
            "Hourly weather data is empty."
        )

    if daily_df.empty:

        raise ValueError(
            "Daily weather data is empty."
        )

    # Convert timestamps
    hourly_df["datetime"] = cast(
        Any,
        pd
    ).to_datetime(
        hourly_df["time"]
    )

    hourly_df["date"] = (
        hourly_df["datetime"]
        .dt.strftime("%Y-%m-%d")
    )

    # Merge daily max/min/wind values
    hourly_df = hourly_df.merge(
        daily_df,
        on="date",
        how="left"
    )

    # Keep exactly the model features
    required = [
        "datetime",
        "T",
        "TM",
        "Tm",
        "SLP",
        "H",
        "VV",
        "V",
        "VM"
    ]

    hourly_df = hourly_df[
        required
    ].copy()

    # Convert model columns to numeric
    for column in FEATURES:

        hourly_df[column] = cast(
            Any,
            pd
        ).to_numeric(
            hourly_df[column],
            errors="coerce"
        )

    return hourly_df


# ============================================================
# FIND WEATHER FOR TARGET HOUR
# ============================================================

def get_target_weather_row(
    weather_df: pd.DataFrame,
    target_datetime: datetime
) -> pd.Series:

    if weather_df.empty:

        raise ValueError(
            "No weather data available."
        )

    # Remove timezone information for comparison
    # because Open-Meteo returns local timestamps
    # when timezone=auto is used.

    target_naive = target_datetime.replace(
        tzinfo=None,
        minute=0,
        second=0,
        microsecond=0
    )

    weather_df = weather_df.copy()

    weather_df["comparison_time"] = (
        weather_df["datetime"]
        .dt.tz_localize(None)
    )

    # Exact hour first
    exact = weather_df[
        weather_df["comparison_time"]
        == target_naive
    ]

    if not exact.empty:

        return exact.iloc[0]

    # If exact hour is unavailable, use nearest hour.
    differences = (
        weather_df["comparison_time"]
        - target_naive
    ).abs()

    nearest_index = differences.idxmin()

    nearest_row = weather_df.loc[[nearest_index]].iloc[0]

    return nearest_row


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_model_input(
    weather_row: pd.Series
) -> pd.DataFrame:

    values: dict[str, float] = {}

    for feature in FEATURES:

        value = weather_row.get(
            feature
        )

        if value is None or value is pd.NA:
            raise ValueError(
                f"Missing model feature: {feature}"
            )

        if isinstance(value, complex):
            raise ValueError(
                f"Missing model feature: {feature}"
            )

        if isinstance(value, (float, int)):
            if value != value:
                raise ValueError(
                    f"Missing model feature: {feature}"
                )

        try:
            values[feature] = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Missing model feature: {feature}"
            ) from None

    dataframe = pd.DataFrame(
        [values],
        columns=FEATURES,
        dtype=float
    )

    return dataframe


# ============================================================
# PREDICT AQI FOR TARGET TIME
# ============================================================

def predict_for_datetime(
    model: Any,
    weather_df: pd.DataFrame,
    target_datetime: datetime
) -> float:

    weather_row = get_target_weather_row(
        weather_df,
        target_datetime
    )

    model_input = build_model_input(
        weather_row
    )

    prediction = float(
        model.predict(model_input)[0]
    )

    # AQI cannot be negative.
    prediction = max(
        0.0,
        prediction
    )

    return round(
        prediction,
        2
    )


# ============================================================
# REVERSE GEOCODING
# ============================================================

def reverse_geocode(
    latitude: float,
    longitude: float
) -> dict[str, str]:

    params: dict[str, Any] = {

        "format": "jsonv2",

        "lat": latitude,

        "lon": longitude,

        "zoom": 18,

        "addressdetails": 1
    }

    try:

        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        address = data.get(
            "address",
            {}
        )

        road = (
            address.get("road")
            or address.get("neighbourhood")
            or address.get("suburb")
            or ""
        )

        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("county")
            or ""
        )

        state = (
            address.get("state")
            or address.get("state_district")
            or ""
        )

        country = (
            address.get("country")
            or ""
        )

        parts: list[str] = []

        for part in [
            road,
            city,
            state,
            country
        ]:

            if (
                part
                and part not in parts
            ):
                parts.append(part)

        name = ", ".join(
            parts
        )

        if not name:

            name = data.get(
                "display_name",
                "Location unavailable"
            )

        return {
            "name": name,
            "city": city,
            "state": state,
            "country": country
        }

    except Exception as error:

        print(
            "Reverse geocoding failed:",
            error
        )

        return {
            "name": "Location unavailable",
            "city": "",
            "state": "",
            "country": ""
        }


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def get_predictions(
    latitude: float,
    longitude: float
) -> dict[str, Any]:

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # LIVE AIR QUALITY
    # --------------------------------------------------------

    air_data = get_current_air_quality(
        latitude,
        longitude
    )

    # --------------------------------------------------------
    # WEATHER FORECAST
    # --------------------------------------------------------

    weather_data = get_weather_forecast(
        latitude,
        longitude
    )

    weather_df = build_weather_dataframe(
        weather_data
    )

    # --------------------------------------------------------
    # DETERMINE CURRENT TIME
    # --------------------------------------------------------

    current_weather_time = (
        weather_df["datetime"]
        .iloc[0]
    )

    current_weather_time = (
        current_weather_time
        .tz_localize(None)
    )

    # --------------------------------------------------------
    # FUTURE TARGET TIMES
    # --------------------------------------------------------

    target_1h = (
        current_weather_time
        + timedelta(hours=1)
    )

    target_24h = (
        current_weather_time
        + timedelta(hours=24)
    )

    target_72h = (
        current_weather_time
        + timedelta(hours=72)
    )

    # --------------------------------------------------------
    # AI PREDICTIONS
    # --------------------------------------------------------

    prediction_1h = predict_for_datetime(
        model,
        weather_df,
        target_1h
    )

    prediction_24h = predict_for_datetime(
        model,
        weather_df,
        target_24h
    )

    prediction_72h = predict_for_datetime(
        model,
        weather_df,
        target_72h
    )

    # --------------------------------------------------------
    # CURRENT AQI
    # --------------------------------------------------------

    current_aqi = float(
        air_data["aqi"]
    )

    current_info = get_aqi_category(
        current_aqi
    )

    info_1h = get_aqi_category(
        prediction_1h
    )

    info_24h = get_aqi_category(
        prediction_24h
    )

    info_72h = get_aqi_category(
        prediction_72h
    )

    # --------------------------------------------------------
    # HEALTH GUIDANCE
    # --------------------------------------------------------

    health_guidance = get_health_guidance(
        current_aqi
    )

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    location = reverse_geocode(
        latitude,
        longitude
    )

    # --------------------------------------------------------
    # CURRENT WEATHER
    # --------------------------------------------------------

    current_weather = get_target_weather_row(
        weather_df,
        current_weather_time
    )

    temperature = float(
        current_weather["T"]
    )

    humidity = float(
        current_weather["H"]
    )

    wind_speed = float(
        current_weather["V"]
    )

    pressure = float(
        current_weather["SLP"]
    )

    visibility = float(
        current_weather["VV"]
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "success": True,

        "predictions": {

            # =================================================
            # LOCATION
            # =================================================

            "location": {

                "name": location["name"],

                "city": location["city"],

                "state": location["state"],

                "country": location["country"],

                "requested_latitude": round(
                    latitude,
                    6
                ),

                "requested_longitude": round(
                    longitude,
                    6
                ),

                "data_latitude": air_data.get(
                    "data_latitude"
                ),

                "data_longitude": air_data.get(
                    "data_longitude"
                ),

                "timezone": air_data.get(
                    "timezone",
                    "Unknown"
                )
            },

            # =================================================
            # TIMESTAMP
            # =================================================

            "updated_at": (
                air_data.get("timestamp")
                or weather_data.get("current", {}).get("time")
            ),

            # =================================================
            # AQI
            # =================================================

            "aqi": {

                "current": {

                    "value": round(
                        current_aqi,
                        2
                    ),

                    **current_info,

                    "source": (
                        "Open-Meteo Current "
                        "Air Quality Data"
                    )
                },

                "1_hour": {

                    "value": prediction_1h,

                    **info_1h,

                    "source": (
                        "AeroPredict Random "
                        "Forest Model"
                    )
                },

                "24_hour": {

                    "value": prediction_24h,

                    **info_24h,

                    "source": (
                        "AeroPredict Random "
                        "Forest Model"
                    )
                },

                "72_hour": {

                    "value": prediction_72h,

                    **info_72h,

                    "source": (
                        "AeroPredict Random "
                        "Forest Model"
                    )
                }
            },

            # =================================================
            # HEALTH
            # =================================================

            "health_guidance": health_guidance,

            # =================================================
            # ENVIRONMENT
            # =================================================

            "environment": {

                "temperature": round(
                    temperature,
                    2
                ),

                "humidity": round(
                    humidity,
                    2
                ),

                "wind_speed": round(
                    wind_speed,
                    2
                ),

                "precipitation": 0.0,

                "pressure": round(
                    pressure,
                    2
                ),

                "visibility": round(
                    visibility,
                    2
                ),

                "timestamp": (
                    weather_df["datetime"]
                    .iloc[0]
                    .isoformat()
                ),

                "source": (
                    "Open-Meteo Weather"
                ),

                "source_type": (
                    "Location-specific "
                    "model-based atmospheric data."
                )
            },

            # =================================================
            # POLLUTANTS
            # =================================================

            "pollutants": {

                "pm2_5": round(
                    air_data["pm2_5"],
                    2
                ),

                "pm10": round(
                    air_data["pm10"],
                    2
                ),

                "nitrogen_dioxide": round(
                    air_data["nitrogen_dioxide"],
                    2
                ),

                "ozone": round(
                    air_data["ozone"],
                    2
                ),

                "sulphur_dioxide": round(
                    air_data["sulphur_dioxide"],
                    2
                ),

                "carbon_monoxide": round(
                    air_data["carbon_monoxide"],
                    2
                )
            },

            # =================================================
            # AI INFORMATION
            # =================================================

            "ai_metadata": {

                "model": (
                    "AeroPredict "
                    "RandomForestRegressor"
                ),

                "prediction_method": (
                    "Random Forest predictions "
                    "generated from future weather "
                    "features."
                ),

                "features_used": len(
                    FEATURES
                ),

                "features": FEATURES,

                "forecast_horizons": [
                    "1 hour",
                    "24 hours",
                    "72 hours"
                ],

                "current_aqi_source": (
                    "Open-Meteo US AQI"
                )
            },

            # =================================================
            # WEATHER METADATA
            # =================================================

            "weather_metadata": {

                "source": (
                    "Open-Meteo"
                ),

                "source_type": (
                    "Location-specific "
                    "forecast weather data."
                ),

                "timestamp": (
                    weather_df["datetime"]
                    .iloc[0]
                    .isoformat()
                )
            }
        }
    }


# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
def root() -> dict[str, Any]:

    return {
        "success": True,
        "application": "AeroPredict",
        "message": (
            "AeroPredict API is running."
        ),
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check() -> dict[str, Any]:

    try:

        model = load_model()

        return {

            "success": True,

            "status": "healthy",

            "model_loaded": True,

            "model_type": (
                type(model).__name__
            ),

            "model_path": MODEL_PATH
        }

    except Exception as error:

        return {

            "success": False,

            "status": "unhealthy",

            "model_loaded": False,

            "error": str(error)
        }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.get("/api/predict")
def predict(

    latitude: float = Query(
        ...,
        description="Latitude of requested location"
    ),

    longitude: float = Query(
        ...,
        description="Longitude of requested location"
    )
):

    # --------------------------------------------------------
    # VALIDATE COORDINATES
    # --------------------------------------------------------

    if not -90 <= latitude <= 90:

        raise HTTPException(
            status_code=400,
            detail=(
                "Latitude must be between "
                "-90 and 90."
            )
        )

    if not -180 <= longitude <= 180:

        raise HTTPException(
            status_code=400,
            detail=(
                "Longitude must be between "
                "-180 and 180."
            )
        )

    # --------------------------------------------------------
    # GENERATE PREDICTION
    # --------------------------------------------------------

    try:

        return get_predictions(
            latitude=latitude,
            longitude=longitude
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        ) from error

    except ValueError as error:

        raise HTTPException(
            status_code=502,
            detail=str(error)
        ) from error

    except requests.RequestException as error:

        raise HTTPException(
            status_code=502,
            detail=(
                "External data service error: "
                f"{error}"
            )
        ) from error

    except Exception as error:

        print(
            "Prediction error:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate prediction: "
                f"{error}"
            )
        ) from error


# ============================================================
# LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print("\n" + "=" * 70)
    print("AEROPREDICT API SERVER")
    print("=" * 70)

    print("\nStarting FastAPI server...")

    print(
        "\nAPI:"
        "\nhttp://127.0.0.1:8001"
    )

    print(
        "\nSwagger documentation:"
        "\nhttp://127.0.0.1:8001/docs"
    )

    print(
        "\nHealth check:"
        "\nhttp://127.0.0.1:8001/health"
    )

    print(
        "\nPrediction endpoint:"
        "\nhttp://127.0.0.1:8001/api/predict"
    )

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        reload=False
    )