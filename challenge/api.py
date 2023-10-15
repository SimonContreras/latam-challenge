"""Fast API module for DelayModel serving on and endpoint: /predict
"""
import os
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

import fastapi
import pandas as pd
from pydantic import BaseModel, ValidationError  # pylint:disable=E0611

from challenge.model import DelayModel

LOG_FMT = "%(levelname)s:   %(asctime)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger(__name__)

app = fastapi.FastAPI()


class Airlines(Enum):
    """Enum class for available Airlines on model"""

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
    """Enum class for Flight types"""

    I = "I"
    N = "N"


class Months(Enum):
    """Month enum as number in str type."""

    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    ELEVEN = "11"
    TWELVE = "12"


class Flight(BaseModel):
    """Pydantic model for API request body contract"""

    OPERA: Airlines
    TIPOVUELO: FlightTypes
    MES: Months


class Flights(BaseModel):
    """Pydantic model for API request body contract"""

    flights: List[Flight]


app = fastapi.FastAPI()


def is_request_valid(data_payload: Dict[str, Any]) -> bool:
    """Check if the body request comply with the allowed
    values for Pydantic model Flights.

    Args:
        data_payload (Dict[str, Any]): Data payload received from request API.

    Returns:
        bool: True if the payload is valid for the corresponding Pydantic model,
        otherwise False.
    """
    is_valid = False
    try:
        Flights.parse_obj(data_payload)
        is_valid = True
    except ValidationError as err:
        logger.error(err)
    return is_valid


async def pickle_fitted_model() -> None:
    """Async method to run on startup event of the API to create the DelayModel
    that will be used during model serving.
    """
    data_path = Path(os.getcwd(), "data/data.csv")
    data = pd.read_csv(filepath_or_buffer=data_path)
    delay_model = DelayModel()
    features_train, target = delay_model.preprocess(data, "delay")
    delay_model.fit(features_train, target)


@app.on_event("startup")
async def startup_event():
    """FastAPI Startup to trigger the DelayModel fit process."""
    logger.info("Fitting model ...")
    await pickle_fitted_model()
    logger.info("Model Fitted!")
    logger.info("FastAPI application has started")


@app.get("/health", status_code=200)
async def get_health() -> Dict[str, str]:
    """Health Endpoint

    Returns:
        dict: A default of response body.
    """
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(
    data_payload: Dict[str, List[Dict[str, str]]]
) -> Dict[str, Union[str, List[int]]]:
    """Predict Endpoint, returns if the flights requested to predict
    will have delay or not delay.

    Args:
        data_payload (Dict[str, List[Dict[str, str]]]): List of flights using the
        structure defined in pydantic model Flights

    Returns:
        Dict[str, Union[str, List[int]]]: Dict with a list of integer with the delay
        prediction for each flight requested, if the request payload is not valid
        returns a error message.
    """
    if not is_request_valid(data_payload):
        return fastapi.responses.JSONResponse(
            {"error": "unknown column received"}, status_code=400
        )
    delay_model = DelayModel()
    request_df = pd.DataFrame(data_payload["flights"])
    features = delay_model.preprocess(request_df)
    return {"predict": delay_model.predict(features)}
