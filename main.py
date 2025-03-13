import requests
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
apikey = os.getenv("API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

def fetch_last_24_hours_weather(city):
    try:
        base_url = "https://api.weatherapi.com/v1/history.json"
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        params = {"key": apikey, "q": city, "dt": yesterday}
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            weather_data = [
                {
                    "dt": hour["time"],
                    "temp": hour["temp_c"],
                    "pressure": hour["pressure_mb"],
                    "humidity": hour["humidity"],
                    "clouds": hour["cloud"],
                    "wind_speed": hour["wind_kph"],
                    "wind_deg": hour["wind_degree"],
                }
                for hour in data["forecast"]["forecastday"][0]["hour"]
            ]
            return {"status": "success", "data": weather_data}

        return {"status": "error", "message": f"Failed to fetch data. Status code: {response.status_code}"}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

def fetch_weather_data(years, latitude, longitude):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    today = datetime.today().date()
    all_data = []

    try:
        for year in range(today.year - years, today.year):
            start_date = datetime(year, today.month, today.day) - timedelta(days=15)
            end_date = datetime(year, today.month, today.day) + timedelta(days=15)

            print(f"Fetching data from {start_date.date()} to {end_date.date()} for year {year}")

            current_date = start_date
            while current_date <= end_date:
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": current_date.strftime("%Y-%m-%d"),
                    "end_date": current_date.strftime("%Y-%m-%d"),
                    "hourly": "temperature_2m,surface_pressure,relative_humidity_2m,cloud_cover,wind_speed_10m,wind_direction_10m",
                    "timezone": "auto",
                }
                response = requests.get(base_url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    for i in range(len(data["hourly"]["time"])):
                        all_data.append([
                            data["hourly"]["time"][i],
                            data["hourly"]["temperature_2m"][i],
                            data["hourly"]["surface_pressure"][i],
                            data["hourly"]["relative_humidity_2m"][i],
                            data["hourly"]["cloud_cover"][i],
                            data["hourly"]["wind_speed_10m"][i],
                            data["hourly"]["wind_direction_10m"][i],
                        ])
                else:
                    print(f"Failed to fetch data for {current_date.strftime('%Y-%m-%d')}")

                current_date += timedelta(days=1)

        print("Weather data fetching complete!")
        return all_data
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")

def preprocess_data(data):
    try:
        df = pd.DataFrame(data, columns=["dt", "temp", "pressure", "humidity", "clouds", "wind_speed", "wind_deg"])
        df["dt"] = pd.to_datetime(df["dt"])
        df.set_index("dt", inplace=True)

        features = ["temp", "pressure", "humidity", "clouds", "wind_speed", "wind_deg"]
        raw_data = df[features].values

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(raw_data)

        return scaler, scaled_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

def create_sequences(data, seq_length=24):
    try:
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating sequences: {str(e)}")
    
def geocode_city(city):
    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    response = requests.get(geocode_url)
    if response.status_code == 200:
        geocode_response = response.json()
        if geocode_response.get("results"):
            latitude = geocode_response["results"][0]["latitude"]
            longitude = geocode_response["results"][0]["longitude"]
            return latitude, longitude
        else:
            raise ValueError("Invalid city name!")
    else:
        raise ConnectionError("Failed to fetch geocoding data.")
    

def train(scaled_data):
    seq_length = 24
    X, y = create_sequences(scaled_data, seq_length)

    X_train = X
    y_train = y

    model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(seq_length, X.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(15, activation='relu'),
            Dense(6),
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    return model

@app.get("/predict/{city}/{years}")
def predict_weather(city: str, years: int):
    try:
        latitude, longitude = geocode_city(city=city)
        all_data = list(fetch_weather_data(years, latitude, longitude))
            
        scaler, scaled_data = preprocess_data(all_data)

        model = train(scaled_data=scaled_data)


        weather_data = fetch_last_24_hours_weather(city)
        X_input = np.array([[d["temp"], d["pressure"], d["humidity"], d["clouds"], d["wind_speed"], d["wind_deg"]] for d in weather_data["data"]])
        X_input = scaler.transform(X_input).reshape(1, 24, 6)

        for i in range(24):
            today_pred = model.predict(X_input)
            X_input = X_input[:, 1:, :]
            today_pred = today_pred.reshape(1, 1, 6)
            X_input = np.append(X_input, today_pred, axis=1) 
    
        today_pred_inv = scaler.inverse_transform(X_input[0])

        predictions = [
            {
                "hour": i,
                "temperature": round(today_pred_inv[i][0])
            } for i in range(len(today_pred_inv))
        ]

        return {"status": "success", "predictions": predictions}


    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


