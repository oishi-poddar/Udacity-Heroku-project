from starlette.testclient import TestClient
from main import app

def test_app_get():
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200
    assert isinstance(response.content,bytes)
    assert str(response.content == "Welcome to my app")


def test_app_prediction():
    dict={
      "workclass": "State-gov",
      "education": "Bachelors",
      "marital_status": "Never-married",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "native_country": "United-States",
      "age": "39",
      "fnlgt": "77516",
      "education_num":"1",
      "capital_gain" : "1",
      "capital_loss": "1",
      "hours_per_week":"10"
        }
    client = TestClient(app)
    response = client.post("/model/", json=dict)
    assert response.status_code == 200
    assert isinstance(response.content, bytes)
    assert list(response.json()) == [0]


def test_app__another_prediction():
    dict={
        "workclass": "Private",
      "education": "Bachelors",
      "marital_status": "Married-civ-spouse",
      "occupation": "Prof-specialty",
      "relationship": "Wife",
      "race": "Black",
      "sex": "Female",
      "native_country": "Cuba",
      "age": 23,
      "fnlgt": 2334,
      "education_num": 7,
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 60
        }
    client = TestClient(app)
    response = client.post("/model/", json=dict)
    assert response.status_code == 200
    assert isinstance(response.content, bytes)
    assert list(response.json()) == [0]