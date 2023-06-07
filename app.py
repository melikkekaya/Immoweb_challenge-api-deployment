from typing import Union, Optional, Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI
from preprocessing.cleaning_data import preprocess
from predict.prediction import predict
import pandas as pd

from fastapi.encoders import jsonable_encoder

app = FastAPI()

class Input(BaseModel):
    area: int
    property_type: Literal ["APARTMENT", "HOUSE"]
    rooms_number: int
    zip_code: int = Field (ge=1000, le=9999)
    land_area: int | None = None
    garden : bool | None = None
    garden_area : int | None = None
    equipped_kitchen: bool | None = None
    swimming_pool: bool | None = None
    furnished: bool | None = None
    open_fire: bool | None = None
    terrace: bool | None = None
    terrace_area: int | None = None
    facades_number: int | None = None
    building_state: Literal ["NEW","GOOD","TO RENOVATE","JUST RENOVATED","TO REBUILD"]

@app.get("/")
async def read_root():
    return "Alive"

@app.get("/predict")
async def data_format():
    return ("""Required informaiton: area: int, property_type:[APARTMENT,HOUSE], 
    rooms_number: int, zip_code: int,land_area: Optional [int], garden : Optional [bool],
    garden_area : Optional [int], equipped_kitchen: Optional [bool], swimming_pool: Optional [bool],
    furnished: Optional [bool], open_fire: Optional [bool], terrace: Optional [bool],
    terrace_area: Optional [int],  facades_number: Optional [int], building_state: [NEW, GOOD,TO RENOVATE,JUST RENOVATED,TO REBUILD]""")


@app.post("/predict") 
def send_prediction(item: Input):
    item_data = jsonable_encoder(item)
    df = preprocess(item_data)
    result = predict(df)
    return result
