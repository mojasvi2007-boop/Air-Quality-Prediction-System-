import pickle
import json
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model1.pkl"

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://nominatim.openstreetmap.org/reverse"

REQUEST_TIMEOUT = 20

MODEL_FEATURES = [
    "T",
    "TM",
    "Tm",
    "SLP",
    "H",
    "VV",
    "V",
    "VM",
]


# ============================================================
# MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_model():
    """Load the reference Random Forest model once."""
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


# ============================================================
# HTTP HELPER
# ============================================================

def fetch_json(
    url: str,
    params: Dict[str, Any] | None = None,
    headers: Dict[str, str] | None = None,
) -> Any:
    """Make a GET request and return JSON."""

    if params:
        url = f"{url}?{urlencode(params)}"

    request = Request(url, headers=headers or {}, method="GET")

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(
            f"HTTP request failed with status {error.code}: {url}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"HTTP request failed: {error.reason}") from error


# ============================================================
# WEATHER DATA
# ============================================================

def get_weather(latitude: float, longitude: float):
    """Get current and forecast weather data."""

    params: Dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,

        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "visibility",
            "wind_speed_10m",
            "surface_pressure",
        ]),

        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "visibility",
            "wind_speed_10m",
            "surface_pressure",
        ]),

        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]),

        "timezone": "auto",
        "forecast_days": 4,
    }

    return fetch_json(WEATHER_URL, params=params)


# ============================================================
# AIR QUALITY DATA
# ============================================================

def get_air_quality(latitude: float, longitude: float):
    """Get current air-quality data."""

    params: dict[str, object] = {
        "latitude": latitude,
        "longitude": longitude,

        "current": ",".join([
            "us_aqi",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ]),

        "hourly": "us_aqi",

        "timezone": "auto",
        "forecast_days": 4,
    }

    return fetch_json(AIR_QUALITY_URL, params=params)


# ============================================================
# LOCATION
# ============================================================

def get_location(latitude: float, longitude: float) -> str:
    """Reverse-geocode coordinates into a readable location."""

    headers = {
        "User-Agent": "AeroPredict/1.0 air-quality-project"
    }

    params: Dict[str, Any] = {
        "lat": latitude,
        "lon": longitude,
        "format": "jsonv2",
        "zoom": 10,
        "addressdetails": 1,
    }

    try:
        data = fetch_json(
            GEOCODING_URL,
            params=params,
            headers=headers,
        )

        address = data.get("address", {})

        location_name = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("county")
            or address.get("state_district")
            or address.get("state")
            or "Unknown location"
        )

        return location_name

    except Exception:
        return "Unknown location"


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi: float):
    """Convert numerical AQI into a US-AQI category."""

    if aqi <= 50:
        return {
            "name": "Good",
            "class": "good",
        }

    if aqi <= 100:
        return {
            "name": "Moderate",
            "class": "moderate",
        }

    if aqi <= 150:
        return {
            "name": "Unhealthy for Sensitive Groups",
            "class": "sensitive",
        }

    if aqi <= 200:
        return {
            "name": "Unhealthy",
            "class": "unhealthy",
        }

    if aqi <= 300:
        return {
            "name": "Very Unhealthy",
            "class": "very-unhealthy",
        }

    return {
        "name": "Hazardous",
        "class": "hazardous",
    }


# ============================================================
# HEALTH ADVICE
# ============================================================

def get_health_advice(aqi: float) -> str:
    """Return general advice based on AQI."""

    if aqi <= 50:
        return "Air quality is good. Normal outdoor activities are suitable."

    if aqi <= 100:
        return "Air quality is acceptable. Sensitive individuals should monitor symptoms."

    if aqi <= 150:
        return (
            "Sensitive individuals should reduce prolonged or heavy outdoor activity."
        )

    if aqi <= 200:
        return (
            "Everyone may begin to experience health effects. "
            "Consider reducing prolonged outdoor activity."
        )

    if aqi <= 300:
        return (
            "Health alert. Avoid prolonged outdoor activity, "
            "especially if you are sensitive to air pollution."
        )

    return (
        "Health emergency conditions. Avoid outdoor exposure "
        "and follow local health guidance."
    )


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_number(value: Any, default: float = 0.0) -> float:
    """Convert a value to a usable float."""

    if value is None:
        return default

    try:
        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


# ============================================================
# MODEL FEATURES
# ============================================================

def build_features(weather: Dict[str, Any], day_index: int = 0):
    """
    Build the exact 8-feature input expected by the
    reference Random Forest model.

    Model order:

    T   = Temperature
    TM  = Maximum Temperature
    Tm  = Minimum Temperature
    SLP = Pressure
    H   = Humidity
    VV  = Visibility
    V   = Wind Speed
    VM  = Maximum Wind Speed
    """

    current = weather.get("current", {})
    daily = weather.get("daily", {})

    temperature = safe_number(
        current.get("temperature_2m")
    )

    humidity = safe_number(
        current.get("relative_humidity_2m")
    )

    visibility_km = (
        safe_number(current.get("visibility")) / 1000.0
    )

    wind_speed = safe_number(
        current.get("wind_speed_10m")
    )

    pressure = safe_number(
        current.get("surface_pressure")
    )

    max_temperatures = daily.get(
        "temperature_2m_max",
        []
    )

    min_temperatures = daily.get(
        "temperature_2m_min",
        []
    )

    max_winds = daily.get(
        "wind_speed_10m_max",
        []
    )

    if day_index >= len(max_temperatures):
        day_index = 0

    if day_index >= len(min_temperatures):
        day_index = 0

    if day_index >= len(max_winds):
        day_index = 0

    max_temperature = safe_number(
        max_temperatures[day_index]
        if max_temperatures
        else temperature
    )

    min_temperature = safe_number(
        min_temperatures[day_index]
        if min_temperatures
        else temperature
    )

    max_wind = safe_number(
        max_winds[day_index]
        if max_winds
        else wind_speed
    )

    features = [
        temperature,
        max_temperature,
        min_temperature,
        pressure,
        humidity,
        visibility_km,
        wind_speed,
        max_wind,
    ]

    return features


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_aqi(model: Any, features: Sequence[float]) -> int:
    """Run the Random Forest prediction."""

    prediction = model.predict(
        np.array(features, dtype=float).reshape(1, -1)
    )[0]

    prediction = safe_number(prediction)

    # Keep AQI inside the standard 0-500 display range.
    prediction = max(0.0, min(500.0, prediction))

    return int(round(prediction))


# ============================================================
# CURRENT ENVIRONMENT
# ============================================================

def get_environment(weather: Dict[str, Any]) -> Dict[str, Any]:
    current = weather.get("current", {})

    return {
        "temperature_c": round(
            safe_number(current.get("temperature_2m")),
            1,
        ),

        "humidity_percent": round(
            safe_number(current.get("relative_humidity_2m")),
            1,
        ),

        "wind_speed_kmh": round(
            safe_number(current.get("wind_speed_10m")),
            1,
        ),

        "precipitation_mm": round(
            safe_number(current.get("precipitation")),
            2,
        ),

        "pressure_hpa": round(
            safe_number(current.get("surface_pressure")),
            1,
        ),

        "visibility_km": round(
            safe_number(current.get("visibility")) / 1000,
            2,
        ),
    }


# ============================================================
# POLLUTANTS
# ============================================================

def get_pollutants(air_quality: Dict[str, Any]) -> Dict[str, float]:
    current = air_quality.get("current", {})

    return {
        "pm2_5": round(
            safe_number(current.get("pm2_5")),
            2,
        ),

        "pm10": round(
            safe_number(current.get("pm10")),
            2,
        ),

        "nitrogen_dioxide": round(
            safe_number(current.get("nitrogen_dioxide")),
            2,
        ),

        "ozone": round(
            safe_number(current.get("ozone")),
            2,
        ),

        "carbon_monoxide": round(
            safe_number(current.get("carbon_monoxide")),
            2,
        ),

        "sulphur_dioxide": round(
            safe_number(current.get("sulphur_dioxide")),
            2,
        ),
    }


# ============================================================
# PREDICTION OBJECT
# ============================================================

def create_prediction_object(aqi: float) -> Dict[str, Any]:
    category = get_aqi_category(aqi)

    return {
        "aqi": aqi,
        "category": category["name"],
        "class": category["class"],
        "advice": get_health_advice(aqi),
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def get_predictions(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Complete AeroPredict pipeline.

    Coordinates
        ↓
    Weather API
        ↓
    Air-quality API
        ↓
    Location API
        ↓
    Random Forest
        ↓
    Current + future predictions
    """

    latitude = safe_number(latitude)
    longitude = safe_number(longitude)

    if not (-90 <= latitude <= 90):
        raise ValueError("Invalid latitude.")

    if not (-180 <= longitude <= 180):
        raise ValueError("Invalid longitude.")

    # Load model
    model = load_model()

    # Fetch external data
    weather = get_weather(
        latitude,
        longitude,
    )

    air_quality = get_air_quality(
        latitude,
        longitude,
    )

    location_name = get_location(
        latitude,
        longitude,
    )

    # --------------------------------------------------------
    # Current AQI
    # --------------------------------------------------------

    current_air_aqi = safe_number(
        air_quality.get("current", {}).get("us_aqi")
    )

    current_air_aqi = int(round(current_air_aqi))

    # --------------------------------------------------------
    # ML Predictions
    # --------------------------------------------------------

    current_features = build_features(
        weather,
        day_index=0,
    )

    prediction_current = predict_aqi(
        model,
        current_features,
    )

    # Future predictions
    prediction_1h = predict_aqi(
        model,
        build_features(weather, 0),
    )

    prediction_24h = predict_aqi(
        model,
        build_features(weather, 1),
    )

    prediction_72h = predict_aqi(
        model,
        build_features(weather, 3),
    )

    return {
        "location": {
            "name": location_name,
            "latitude": latitude,
            "longitude": longitude,
        },

        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "current": {
            "aqi": current_air_aqi,
            "category": get_aqi_category(
                current_air_aqi
            )["name"],
            "class": get_aqi_category(
                current_air_aqi
            )["class"],
            "advice": get_health_advice(
                current_air_aqi
            ),
            "source": "Open-Meteo",
        },

        "ml_prediction": {
            "aqi": prediction_current,
            "category": get_aqi_category(
                prediction_current
            )["name"],
            "class": get_aqi_category(
                prediction_current
            )["class"],
            "advice": get_health_advice(
                prediction_current
            ),
        },

        "forecast": {
            "1_hour": create_prediction_object(
                prediction_1h
            ),

            "24_hour": create_prediction_object(
                prediction_24h
            ),

            "72_hour": create_prediction_object(
                prediction_72h
            ),
        },

        "environment": get_environment(
            weather
        ),

        "pollutants": get_pollutants(
            air_quality
        ),

        "model": {
            "name": "Random Forest Regression",
            "type": "Machine Learning",
            "features": MODEL_FEATURES,
            "feature_count": len(MODEL_FEATURES),
            "status": "live",
        },

        "data_sources": {
            "weather": "Open-Meteo",
            "air_quality": "Open-Meteo / CAMS",
            "geocoding": "OpenStreetMap Nominatim",
        },
    }