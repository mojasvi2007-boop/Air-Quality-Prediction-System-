/* =========================================================
   AeroPredict - Frontend JavaScript
   ========================================================= */

const API_URL = "http://127.0.0.1:8001";

let map = null;
let marker = null;
let accuracyCircle = null;


/* =========================================================
   HELPER FUNCTIONS
   ========================================================= */

function $(id) {
    return document.getElementById(id);
}


function setText(id, value) {
    const element = $(id);

    if (element) {
        element.textContent = value;
    } else {
        console.warn(`Element #${id} not found`);
    }
}


function formatNumber(value, decimals = 1) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(Number(value))
    ) {
        return "--";
    }

    return Number(value).toFixed(decimals);
}


/* =========================================================
   AQI CATEGORY CLASS
   ========================================================= */

function getAQIClass(category) {

    const text =
        String(category || "").toLowerCase();

    if (text.includes("hazardous")) {
        return "hazardous";
    }

    if (text.includes("very unhealthy")) {
        return "very-unhealthy";
    }

    if (text.includes("unhealthy")) {
        return "unhealthy";
    }

    if (text.includes("sensitive")) {
        return "unhealthy-sensitive";
    }

    if (
        text.includes("moderate") ||
        text.includes("satisfactory")
    ) {
        return "moderate";
    }

    return "good";
}


/* =========================================================
   LOCATION
   ========================================================= */

function displayLocation(
    latitude,
    longitude,
    accuracy,
    name
) {

    setText(
        "locationName",
        name || "Your Location"
    );


    if (
        latitude !== null &&
        latitude !== undefined &&
        longitude !== null &&
        longitude !== undefined
    ) {

        setText(
            "coordinates",
            `${Number(latitude).toFixed(5)}, ${Number(longitude).toFixed(5)}`
        );
    }


    if (
        accuracy !== null &&
        accuracy !== undefined &&
        Number.isFinite(Number(accuracy))
    ) {

        const accuracyValue =
            Number(accuracy);

        if (accuracyValue >= 1000) {

            setText(
                "accuracy",
                `${(accuracyValue / 1000).toFixed(1)} km accuracy`
            );

        } else {

            setText(
                "accuracy",
                `${Math.round(accuracyValue)} m accuracy`
            );
        }

    } else {

        setText(
            "accuracy",
            "Location accuracy unavailable"
        );
    }
}


/* =========================================================
   ENVIRONMENT
   ========================================================= */

function displayEnvironment(environment) {

    console.log(
        "ENVIRONMENT DATA:",
        environment
    );


    if (!environment) {

        console.error(
            "Environment data is missing."
        );

        return;
    }


    /*
       NEW FASTAPI STRUCTURE:

       environment.temperature
       environment.humidity
       environment.wind_speed
       environment.precipitation
       environment.pressure
       environment.visibility
    */


    const temperature =
        environment.temperature;

    const humidity =
        environment.humidity;

    const wind =
        environment.wind_speed;


    setText(
        "temperature",
        temperature !== null &&
        temperature !== undefined
            ? `${formatNumber(temperature)} °C`
            : "--"
    );


    setText(
        "humidity",
        humidity !== null &&
        humidity !== undefined
            ? `${formatNumber(humidity)} %`
            : "--"
    );


    setText(
        "wind",
        wind !== null &&
        wind !== undefined
            ? `${formatNumber(wind)} km/h`
            : "--"
    );
}


/* =========================================================
   POLLUTANTS
   ========================================================= */

function displayPollutants(pollutants) {

    console.log(
        "POLLUTANT DATA:",
        pollutants
    );


    if (!pollutants) {

        console.error(
            "Pollutant data is missing."
        );

        return;
    }


    setText(
        "pm25",
        pollutants.pm2_5 !== null &&
        pollutants.pm2_5 !== undefined
            ? formatNumber(pollutants.pm2_5)
            : "--"
    );


    setText(
        "pm10",
        pollutants.pm10 !== null &&
        pollutants.pm10 !== undefined
            ? formatNumber(pollutants.pm10)
            : "--"
    );


    setText(
        "no2",
        pollutants.nitrogen_dioxide !== null &&
        pollutants.nitrogen_dioxide !== undefined
            ? formatNumber(
                pollutants.nitrogen_dioxide
            )
            : "--"
    );


    setText(
        "o3",
        pollutants.ozone !== null &&
        pollutants.ozone !== undefined
            ? formatNumber(
                pollutants.ozone
            )
            : "--"
    );


    setText(
        "so2",
        pollutants.sulphur_dioxide !== null &&
        pollutants.sulphur_dioxide !== undefined
            ? formatNumber(
                pollutants.sulphur_dioxide
            )
            : "--"
    );


    setText(
        "co",
        pollutants.carbon_monoxide !== null &&
        pollutants.carbon_monoxide !== undefined
            ? formatNumber(
                pollutants.carbon_monoxide
            )
            : "--"
    );
}


/* =========================================================
   AQI
   ========================================================= */

function displayAQI(
    aqi,
    category,
    advice
) {

    console.log(
        "CURRENT AQI:",
        aqi,
        category
    );


    setText(
        "aqiValue",
        aqi !== null &&
        aqi !== undefined
            ? Math.round(Number(aqi))
            : "--"
    );


    setText(
        "aqiCategory",
        category || "--"
    );


    setText(
        "aqiAdvice",
        advice ||
        "Air-quality information is being calculated."
    );


    const aqiValueElement =
        $("aqiValue");

    const aqiCategoryElement =
        $("aqiCategory");


    const classes = [
        "good",
        "moderate",
        "unhealthy-sensitive",
        "unhealthy",
        "very-unhealthy",
        "hazardous"
    ];


    if (aqiValueElement) {

        aqiValueElement.classList.remove(
            ...classes
        );

        aqiValueElement.classList.add(
            getAQIClass(category)
        );
    }


    if (aqiCategoryElement) {

        aqiCategoryElement.classList.remove(
            ...classes
        );

        aqiCategoryElement.classList.add(
            getAQIClass(category)
        );
    }
}


/* =========================================================
   FORECAST
   ========================================================= */

function displayForecast(
    valueId,
    categoryId,
    forecast
) {

    console.log(
        `FORECAST ${valueId}:`,
        forecast
    );


    if (!forecast) {

        setText(
            valueId,
            "--"
        );

        setText(
            categoryId,
            "--"
        );

        return;
    }


    /*
       NEW FASTAPI STRUCTURE:

       {
           value: 80.97,
           category: "Moderate",
           description: "...",
           source: "AeroPredict Random Forest Model"
       }
    */


    const aqi =
        forecast.value;

    const category =
        forecast.category;


    setText(
        valueId,
        aqi !== null &&
        aqi !== undefined
            ? Math.round(Number(aqi))
            : "--"
    );


    setText(
        categoryId,
        category || "--"
    );


    const valueElement =
        $(valueId);

    const categoryElement =
        $(categoryId);


    const classes = [
        "good",
        "moderate",
        "unhealthy-sensitive",
        "unhealthy",
        "very-unhealthy",
        "hazardous"
    ];


    if (valueElement) {

        valueElement.classList.remove(
            ...classes
        );

        valueElement.classList.add(
            getAQIClass(category)
        );
    }


    if (categoryElement) {

        categoryElement.classList.remove(
            ...classes
        );

        categoryElement.classList.add(
            getAQIClass(category)
        );
    }
}


/* =========================================================
   MAP
   ========================================================= */

function updateMap(
    latitude,
    longitude,
    accuracy
) {

    if (typeof L === "undefined") {

        console.warn(
            "Leaflet has not loaded."
        );

        return;
    }


    const lat =
        Number(latitude);

    const lon =
        Number(longitude);


    if (
        Number.isNaN(lat) ||
        Number.isNaN(lon)
    ) {

        console.error(
            "Invalid map coordinates."
        );

        return;
    }


    if (!map) {

        map =
            L.map("map", {
                zoomControl: true
            }).setView(
                [lat, lon],
                13
            );


        L.tileLayer(
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            {
                attribution:
                    "&copy; OpenStreetMap contributors"
            }
        ).addTo(map);

    } else {

        map.setView(
            [lat, lon],
            13
        );
    }


    if (marker) {

        map.removeLayer(
            marker
        );
    }


    if (accuracyCircle) {

        map.removeLayer(
            accuracyCircle
        );
    }


    marker =
        L.marker(
            [lat, lon]
        ).addTo(map);


    marker.bindPopup(
        "<strong>Your detected location</strong>"
    );


    if (
        accuracy !== null &&
        accuracy !== undefined &&
        Number(accuracy) > 0
    ) {

        accuracyCircle =
            L.circle(
                [lat, lon],
                {
                    radius:
                        Number(accuracy)
                }
            ).addTo(map);
    }


    setTimeout(
        () => {

            map.invalidateSize();

        },
        200
    );
}


/* =========================================================
   RENDER API RESPONSE
   ========================================================= */

function renderPrediction(
    result,
    latitude,
    longitude,
    accuracy
) {

    console.log(
        "FULL API RESPONSE:",
        result
    );


    if (!result) {

        throw new Error(
            "Empty response received from API."
        );
    }


    /*
       CURRENT FASTAPI RESPONSE:

       result
          └── predictions
                ├── location
                ├── updated_at
                ├── aqi
                │     ├── current
                │     ├── 1_hour
                │     ├── 24_hour
                │     └── 72_hour
                ├── health_guidance
                ├── environment
                ├── pollutants
                ├── ai_metadata
                └── weather_metadata
    */


    const data =
        result.predictions ||
        result.data ||
        result;


    if (!data) {

        throw new Error(
            "Prediction data not found."
        );
    }


    /* -------------------------------------------------------
       LOCATION
       ------------------------------------------------------- */

    const location =
        data.location || {};


    displayLocation(
        latitude,
        longitude,
        accuracy,
        location.name ||
        "Your Location"
    );


    /* -------------------------------------------------------
       ENVIRONMENT
       ------------------------------------------------------- */

    displayEnvironment(
        data.environment
    );


    /* -------------------------------------------------------
       POLLUTANTS
       ------------------------------------------------------- */

    displayPollutants(
        data.pollutants
    );


    /* -------------------------------------------------------
       CURRENT AQI
       ------------------------------------------------------- */

    const aqi =
        data.aqi || {};


    const currentAQI =
        aqi.current || {};


    displayAQI(
        currentAQI.value,
        currentAQI.category,
        data.health_guidance?.message
    );


    /* -------------------------------------------------------
       AI FORECAST
       ------------------------------------------------------- */

    displayForecast(
        "forecast1",
        "forecast1Category",
        aqi["1_hour"]
    );


    displayForecast(
        "forecast24",
        "forecast24Category",
        aqi["24_hour"]
    );


    displayForecast(
        "forecast72",
        "forecast72Category",
        aqi["72_hour"]
    );


    /* -------------------------------------------------------
       MAP
       ------------------------------------------------------- */

    updateMap(
        latitude,
        longitude,
        accuracy
    );
}


/* =========================================================
   API REQUEST
   ========================================================= */

async function fetchPrediction(
    latitude,
    longitude,
    accuracy
) {

    const statusElement =
        $("status");

    const button =
        $("locationBtn");


    try {

        if (button) {

            button.disabled =
                true;

            button.textContent =
                "Loading Air Quality...";
        }


        if (statusElement) {

            statusElement.textContent =
                "Fetching live air-quality data...";
        }


        const url =
            `${API_URL}/api/predict` +
            `?latitude=${encodeURIComponent(latitude)}` +
            `&longitude=${encodeURIComponent(longitude)}`;


        console.log(
            "Requesting:",
            url
        );


        const response =
            await fetch(url);


        if (!response.ok) {

            let errorMessage =
                `API request failed: HTTP ${response.status}`;


            try {

                const errorData =
                    await response.json();

                errorMessage =
                    errorData.detail ||
                    errorMessage;

            } catch (error) {

                console.warn(
                    "Could not parse API error response."
                );
            }


            throw new Error(
                errorMessage
            );
        }


        const result =
            await response.json();


        console.log(
            "API RESULT:",
            result
        );


        if (
            result.success === false
        ) {

            throw new Error(
                result.message ||
                result.detail ||
                "Prediction failed."
            );
        }


        renderPrediction(
            result,
            latitude,
            longitude,
            accuracy
        );


        if (statusElement) {

            statusElement.textContent =
                "Live air-quality data loaded.";
        }


        if (button) {

            button.textContent =
                "Refresh My Location";
        }

    }

    catch (error) {

        console.error(
            "AeroPredict error:",
            error
        );


        if (statusElement) {

            statusElement.textContent =
                `Unable to load prediction: ${error.message}`;
        }


        if (button) {

            button.textContent =
                "Try Again";
        }
    }

    finally {

        if (button) {

            button.disabled =
                false;
        }
    }
}


/* =========================================================
   GEOLOCATION
   ========================================================= */

function detectLocation() {

    const statusElement =
        $("status");

    const button =
        $("locationBtn");


    if (!navigator.geolocation) {

        if (statusElement) {

            statusElement.textContent =
                "Geolocation is not supported by this browser.";
        }

        return;
    }


    if (button) {

        button.disabled =
            true;

        button.textContent =
            "Detecting Location...";
    }


    if (statusElement) {

        statusElement.textContent =
            "Detecting your location...";
    }


    navigator.geolocation.getCurrentPosition(

        async function(position) {

            const latitude =
                position.coords.latitude;

            const longitude =
                position.coords.longitude;

            const accuracy =
                position.coords.accuracy;


            console.log(
                "Browser location:",
                {
                    latitude,
                    longitude,
                    accuracy
                }
            );


            displayLocation(
                latitude,
                longitude,
                accuracy,
                "Detecting..."
            );


            await fetchPrediction(
                latitude,
                longitude,
                accuracy
            );
        },


        function(error) {

            console.error(
                "Geolocation error:",
                error
            );


            let message =
                "Unable to detect your location.";


            if (error.code === 1) {

                message =
                    "Location permission was denied.";

            } else if (error.code === 2) {

                message =
                    "Your location could not be determined.";

            } else if (error.code === 3) {

                message =
                    "Location request timed out.";
            }


            if (statusElement) {

                statusElement.textContent =
                    message;
            }


            if (button) {

                button.disabled =
                    false;

                button.textContent =
                    "Try Again";
            }
        },


        {
            enableHighAccuracy: true,
            timeout: 30000,
            maximumAge: 0
        }
    );
}


/* =========================================================
   BUTTON
   ========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    function() {

        const locationButton =
            $("locationBtn");


        if (!locationButton) {

            console.error(
                "Could not find #locationBtn"
            );

            return;
        }


        locationButton.addEventListener(
            "click",
            detectLocation
        );


        console.log(
            "AeroPredict frontend initialized."
        );
    }
);