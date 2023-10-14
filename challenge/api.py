import os
from enum import Enum
from pathlib import Path
from typing import Dict, List

import fastapi
import pandas as pd
from pydantic import BaseModel

from challenge.model import DelayModel


class Airlines(Enum):
    AEROLINEAS_ARGENTINAS = "Aerolineas Argentinas"
    AEROMEXICO = "Aeromexico"
    AIR_CANADA = "Air Canada"
    AIR_FRANCE = "Air France"
    ALITALIA = "Alitalia"
    AMERICAN_AIRLINES = "American Airlines"
    AUSTRAL = "Austral"
    AVIANCA = "Avianca"
    BRITISH_AIRWAYS = "British Airways"
    COPA_AIR = "Copa Air"
    DELTA_AIR = "Delta Air"
    GOL_TRANS = "Gol Trans"
    GRUPO_LATAM = "Grupo LATAM"
    IBERIA = "Iberia"
    JETSMART_SPA = "JetSmart SPA"
    K_L_M = "K.L.M."
    LACSA = "Lacsa"
    LATIN_AMERICAN_WINGS = "Latin American Wings"
    OCEANAIR_LINHAS_AEREAS = "Oceanair Linhas Aereas"
    PLUS_ULTRA_LINEAS_AEREAS = "Plus Ultra Lineas Aereas"
    QANTAS_AIRWAYS = "Qantas Airways"
    SKY_AIRLINE = "Sky Airline"
    UNITED_AIRLINES = "United Airlines"


class FlightTypes(Enum):
    I = "I"
    N = "N"


class Months(Enum):
    one = "1"
    two = "2"
    three = "3"
    four = "4"
    five = "5"
    six = "6"
    seven = "7"
    eight = "8"
    nine = "9"
    ten = "10"
    eleven = "11"
    twelve = "12"


class Flight(BaseModel):
    OPERA: Airlines
    TIPOVUELO: FlightTypes
    MES: Months


class Flights(BaseModel):
    flights: List[Flight]


app = fastapi.FastAPI()


async def pickle_fitted_model():
    data_path = Path(os.getcwd(), "data/data.csv")
    data = pd.read_csv(filepath_or_buffer=data_path)
    delay_model = DelayModel()
    features_train, target = delay_model.preprocess(data, "delay")
    delay_model.fit(features_train, target)


@app.on_event("startup")
async def startup_event():
    await pickle_fitted_model()
    print("FastAPI application has started")


# Define your API routes and other FastAPI components below


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(data_payload: Dict[str, List[Dict[str, str]]]) -> dict:
    try:
        Flights.parse_obj(data_payload)
    except Exception:
        return fastapi.responses.JSONResponse(
            {"error": "unknown column received"}, status_code=400
        )
    delay_model = DelayModel()
    request_df = pd.DataFrame(data_payload["flights"])
    features = delay_model.preprocess(request_df)
    return {"predict": delay_model.predict(features)}
