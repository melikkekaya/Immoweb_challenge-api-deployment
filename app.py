from typing import Union, Optional, Literal
from pydantic import BaseModel
from fastapi import FastAPI
from preprocessing.cleaning_data import preprocess
from predict.prediction import predict
import pandas as pd

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
    full_address: Optional[str]
    swimming_pool: bool | None = None
    furnished: bool | None = None
    open_fire: bool | None = None
    terrace: bool | None = None
    terrace_area: int | None = None
    facades_number: int | None = None
    building_state: Literal["NEW","GOOD","TO RENOVATE","JUST RENOVATED","TO REBUILD"]

@app.get("/")
async def read_root():
    return "Alive"


json_data = {
"area": 80,
"property-type": "HOUSE",
"rooms-number" : 4,
"zip-code": 1000,
"land-area": 180,
"garden": 1,
"garden-area": 50,
"equipped-kitchen": True,
"full-address": "dfkgfg",
"swimming-pool": 0,
"furnished": False,
"open-fire": 0,
"terrace": 0,
"terrace-area": 0,
"facades-number": 4,
"building-state": "NEW"
}
# df = preprocess(json_data)
# result = predict(df)
# print(result)


@app.post("/predict")
async def send_prediction(item: Input):
    df = preprocess(item)
    result = predict(df)
    # result = item.zip_code
    return {"result": result[0]}

# info = dict1.values[0]
# print(info)


# model = LogisticRegression()
# model.fit(X_train, Y_train)

# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
 
# # some time later...
 
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)