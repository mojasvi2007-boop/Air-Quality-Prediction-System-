 # AeroPredict

AI-powered air quality prediction system using machine learning, live weather data, air quality data, and browser-based geolocation.

## Overview

AeroPredict is a web-based air quality prediction system that combines environmental data with a trained Random Forest regression model to estimate future air quality conditions.

The application retrieves the user's geographic location through the browser, collects current and forecast weather and air quality data, and presents the results through an interactive dashboard.

## Features

- Live browser-based geolocation
- Reverse geocoding for location identification
- Current air quality information
- PM2.5, PM10, NO₂, O₃, SO₂ and CO data
- Temperature, humidity, wind speed, pressure and visibility
- AI-based air quality predictions for:
  - 1 hour
  - 24 hours
  - 72 hours
- Interactive map
- Random Forest regression model
- FastAPI backend
- Responsive web dashboard

## Machine Learning

The prediction model is a `RandomForestRegressor` trained using environmental and weather-related features.

### Model Features

The model uses eight input features:

| Feature | Description |
|---|---|
| T | Temperature |
| TM | Maximum Temperature |
| Tm | Minimum Temperature |
| SLP | Sea-Level Pressure |
| H | Humidity |
| VV | Visibility |
| V | Wind Speed |
| VM | Maximum Wind Speed |

The trained model is stored at:

```text
models/model1.pkl