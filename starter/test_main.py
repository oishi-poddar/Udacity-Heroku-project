from starlette.testclient import TestClient
from main import app

def test_app_get():
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200
    assert isinstance(response.content,bytes)
    assert response.content == "Welcome to my app"

def test_app_prediction_below_50k():
    dictionary = {
      "age": 39,
      "fnlgt": 77516,
      "education_num": 13,
      "capital_gain": 2174,
      "capital_loss": 0,
      "hours_per_week": 40,
      "workclass": "State-gov",
      "education": "Bachelors",
      "marital_status": "Never-married",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "native_country": "United-States"
    }
    client = TestClient(app)
    response = client.post("/model/", json=dictionary)
    assert response.status_code == 200
    assert isinstance(response.content, bytes)
    assert list(response.json()) == [0]  #  testing one possible case where resturn value is 0 signifying salary <=50k

def test_app_prediction_above_50k():
    dictionary={
        "age": 52,
        "fnlgt": 209642,
        "education_num": 9,
        "capital_gain": 123387,
        "capital_loss": 0,
        "hours_per_week": 40,
        "workclass": "Self-emp-not-inc",
        "education": "Bachelors",
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native_country": "United-States"
    }
    client = TestClient(app)
    response = client.post("/model/", json=dictionary)
    assert response.status_code == 200
    assert isinstance(response.content, bytes)
    assert list(response.json()) == [1]  #  testing another possible case where return value is 1 signifying salary above 50k