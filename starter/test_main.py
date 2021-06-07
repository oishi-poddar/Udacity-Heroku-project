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
      "workclass": "Self-emp-not-inc",
      "education": "HS-grad",
      "marital_status": "Married-civ-spouse",
      "occupation": "Exec-managerial",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "native_country": "United-States",
      "age": "39",
      "fnlgt": "77516",
      "education_num":"9",
      "capital_gain" : "1",
      "capital_loss": "1",
      "hours_per_week":"10"
        }
    client = TestClient(app)
    response = client.post("/model/", json=dict)
    assert response.status_code == 200
    assert isinstance(response.content, bytes)
    assert list(response.json()) == [1]  #  testing one possible case where resturn value is 1


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
    assert list(response.json()) == [0]   #  testing another possible case where return value is 0