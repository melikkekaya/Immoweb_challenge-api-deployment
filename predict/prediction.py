import pickle
import pandas as pd
from preprocessing.cleaning_data import preprocess

import numpy as np
import sklearn

with open("./model/model_house.sav", 'rb') as file:
    house_model = pickle.load(file)

with open("./model/model_apt.sav", 'rb') as file:
    apt_model = pickle.load(file)

def predict(df):
    if df["Type"].values[0] == "HOUSE":
        df = df.loc[:, ~df.columns.isin(["Type"])]
        return house_model.predict(df)
    
    elif df["Type"].values[0] ==  "APARTMENT":
        df = df.loc[:, ~df.columns.isin(["Type","Surface of the land","Number of facades"])]
        return apt_model.predict(df)