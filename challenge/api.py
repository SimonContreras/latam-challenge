import os
from pathlib import Path
from typing import Dict, List

import fastapi
import pandas as pd

from challenge.model import DelayModel

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
    request_df = pd.DataFrame(data_payload["flights"])
    delay_model = DelayModel()
    # if not DelayModel.are_columns_valid(list(request_df.columns)):
    # return fastapi.responses.JSONResponse(
    #    {"error": "unknown column received"}, status_code=400
    # )
    # crear clase pydantic para validar input si tira validation error, retorno 400
    request_df = pd.DataFrame(data_payload["flights"])
    features = delay_model.preprocess(request_df)
    return {"predict": delay_model.predict(features)}
