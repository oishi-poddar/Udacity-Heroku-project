import requests
import json

url = 'https://oishi-herokuapp.herokuapp.com/model'
myobj = {
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
print("sending request via POST to url", url)
headers = {'Content-type': 'application/json'}
x = requests.post(url, data = json.dumps(myobj), headers=headers)
print("Response is", x.text)
print("Response status code is", x.status_code)
