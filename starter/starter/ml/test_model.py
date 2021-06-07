import pickle
import numpy as np
from starter.starter.ml.model import inference, train_model, compute_model_metrics
import os
import sklearn.naive_bayes

def test_inference():
    filename = 'startertrainedmodel' + '.pkl'
    path = os.path.join(os.getcwd() + "/starter/starter/ml")
    with open(path + "/" +filename, 'rb') as file:
        model = pickle.load(file)
    X = np.array([np.zeros(108)])
    assert isinstance(inference(model,X), np.ndarray)


def test_train_model():
    X= np.array([[1, 2, 3], [4, 5, 6],[1, 2, 3], [4, 5, 6]])
    Y= np.array([0, 1,0,1])
    model = train_model(X,Y)
    assert type(model) == sklearn.naive_bayes.GaussianNB

def test_compute_model_metrics():
    y = np.zeros(5)
    preds = np.zeros(5)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
