# Put the code for your API here.

from pydantic import BaseModel
from starter.ml.model import inference
from starter.ml.data import process_data
from starter.train_model import train
import pickle
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, status
from fastapi.responses import JSONResponse
import json
app = FastAPI()

@app.get("/")
async def root():
    print("here")

class Input(BaseModel):
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    age: int
    fnlgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int

@app.post("/model/")
async def post_train_model(input: Input):
    filename = 'trainedmodel' + '.pkl'
    with open("starter/starter" + filename, 'rb') as file:
        model = pickle.load(file)
    with open("encoder.pickle", 'rb') as file:
        encoder = pickle.load(file)
    with open("lib.pickle", 'rb') as file:
        lb = pickle.load(file)
    dict={ "workclass": input.workclass,
            "education": input.education,
            "marital-status": input.marital_status,
            "occupation": input.occupation,
            "relationship": input.relationship,
            "race": input.race,
            "sex": input.sex,
            "native-country": input.native_country,
            "age": input.age,
            "fnlgt": input.fnlgt,
            "education-num": input.education_num,
           "capital-gain" : input.capital_gain,
           "capital-loss": input.capital_loss,
           "hours-per-week": input.hours_per_week

           }
    columns = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
            "age",
            "fnlgt",
             "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week"
        ]
    features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country"
        ]


    # transform input to a dataframe
    df = pd.DataFrame(dict, columns=columns,  index=[0])
    X, y, encoder, lb = process_data(
        df, categorical_features=features, label=None, training=False,
        encoder=encoder, lb= lb
    )
    preds = inference(model, X)
    return JSONResponse(status_code=200, content=preds.tolist())
