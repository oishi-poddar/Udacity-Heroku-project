# Put the code for your API here.

from pydantic import BaseModel, Field
from starter.ml.model import inference
from starter.ml.data import process_data
import pickle
import pandas as pd
from fastapi import Body, FastAPI, status
from fastapi.responses import JSONResponse
import os

app = FastAPI()

@app.get("/")
async def root():
    return JSONResponse(status_code=200, content="Welcome to my app")

class Input(BaseModel):
    workclass: str = Field(..., example="Private")
    education: str = Field(..., example="Bachelors")
    marital_status: str = Field(..., example="Married-civ-spouse")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Wife")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Female")
    native_country: str = Field(..., example="Cuba")
    age: int = Field(..., example=23)
    fnlgt: int = Field(..., example=2334)
    education_num: int = Field(..., example=7)
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=60)

@app.post("/model/")
async def post_train_model(input: Input):
    filename = 'startertrainedmodel' + '.pkl'
    path = os.path.join(os.getcwd()+"/starter/starter/ml")
    with open(path+"/"+filename, 'rb') as file:
        model = pickle.load(file)
    path = os.path.join(os.getcwd() + "/starter")
    with open(path + "/encoder.pickle", 'rb') as file:
        encoder = pickle.load(file)
    with open(path + "/lib.pickle", 'rb') as file:
        lb = pickle.load(file)
    dict = input.dict()
    columns = dict.keys()
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
