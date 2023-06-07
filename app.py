from typing import Union, Optional, Literal
from pydantic import BaseModel
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
    zip_code: int
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

json_data = {
"area": 80,
"property_type": "HOUSE",
"rooms_number" : 4,
"zip_code": 1000,
"land_area": 180,
"garden": True,
"garden_area": 50,
"equipped_kitchen": True,
"swimming_pool": 0,
"furnished": False,
"open_fire": 0,
"terrace": 0,
"terrace_area": 0,
"facades_number": 4,
"building_state": "NEW"
}

@app.post("/predict")
def send_prediction(item: Input):

    json_compatible_item_data = jsonable_encoder(item)
    # print ("item type: ", type(item))
    # print("json_compatible_item_data: " , type(json_compatible_item_data))
    df = preprocess(json_compatible_item_data)
  
  
    # df = preprocess(item)
    result = predict(df)
    return {"result": result[0]}

# print(send_prediction(json_data))