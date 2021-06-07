from starlette.testclient import TestClient
from main import app

def test_app_get():
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200
    assert isinstance(response.content,bytes)
    assert str(response.content == "Welcome to my app")

def test_app_prediction_50k_below():
    dictionary = {
      "age": 39,
      "fnlgt": 77516,
      "education-num": 13,
      "capital-gain": 2174,
      "capital-loss": 0,
      "hours-per-week": 40,
      "workclass": "State-gov",
      "education": "Bachelors",
      "marital-status": "Never-married",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "native-country": "United-States"
    }
    client = TestClient(app)
    response = client.post("/model/", json=dictionary)
    assert response.status_code == 200
    assert isinstance(response.content, bytes)
    # assert list(response.json()) == [1]  #  testing one possible case where resturn value is 1
    assert response.text == '\"<=50K\"'

def test_app__another_prediction_above_50k():
    dictionary={
        "age": 52,
        "fnlgt": 209642,
        "education-num": 9,
        "capital-gain": 123387,
        "capital-loss": 0,
        "hours-per-week": 40,
        "workclass": "Self-emp-not-inc",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States"
    }
    client = TestClient(app)
    response = client.post("/model/", json=dictionary)
    assert response.status_code == 200
    assert isinstance(response.content, bytes)
    # assert list(response.json()) == [0]   #  testing another possible case where return value is 0
    assert response.text == '\">50K\"'