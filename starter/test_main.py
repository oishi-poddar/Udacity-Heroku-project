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
      "education": "Bachelors",
      "marital_status": "Married-civ-spouse",
      "occupation": "Exec-managerial",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "native_country": "United-States",
      "age": "52",
      "fnlgt": "209642",
      "education_num":"9",
      "capital_gain" : "0",
      "capital_loss": "0",
      "hours_per_week":"45"
        }
    client = TestClient(app)
    response = client.post("/model/", json=dict)
    assert response.status_code == 200
    assert isinstance(response.content, bytes)
    # assert list(response.json()) == [1]  #  testing one possible case where resturn value is 1
    assert response.text == '\">50K\"'

def test_app__another_prediction():
    dict={
        "workclass": "Private",
      "education": "9th",
      "marital_status": "Married-spouse-absent",
      "occupation": "Other-service",
      "relationship": "Not-in-family",
      "race": "Black",
      "sex": "Female",
      "native_country": "Cuba",
      "age": 49,
      "fnlgt": 160187,
      "education_num": 5,
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 16
        }
    client = TestClient(app)
    response = client.post("/model/", json=dict)
    assert response.status_code == 200
    assert isinstance(response.content, bytes)
    # assert list(response.json()) == [0]   #  testing another possible case where return value is 0
    assert response.text == '\"<=50K\"'