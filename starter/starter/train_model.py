# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
# from starter.ml.data import process_data
# from starter.ml.model import train_model, inference, compute_model_metrics
import pickle
import os

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
# Add the necessary imports for the starter code.

# Add code to load in the data.
def train():
    path = os.path.join(os.getcwd() + "/starter/data")
    filename = "clean_census_data.csv"
    data = pd.read_csv(path + "/" +filename)

    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    # Train and save a model
    model = train_model(X_train, y_train)
    # save model, encoder and lb to pickle file
    filename = 'startertrainedmodel' + '.pkl'
    path = os.path.join(os.getcwd() + "/starter/starter/ml")
    with open(path + '/' +filename, 'wb') as file:
        pickle.dump(model, file)
    path = os.path.join(os.getcwd() + "/starter")
    with open(path + '/encoder.pickle', 'wb') as file:
        pickle.dump(encoder, file)
    with open(path + '/lib.pickle', 'wb') as f:
        pickle.dump(lb, f)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    return model

def fetch_file(path, filename):
    with open(path + "/" + filename, 'rb') as file:
        file = pickle.load(file)
    return file

# performance testing using model slicing
def model_slicing(data, cat_features):

    path = os.path.join(os.getcwd() + "/starter/starter/ml")
    model = fetch_file(path, filename='startertrainedmodel' + '.pkl')
    path = os.path.join(os.getcwd() + "/starter")
    encoder = fetch_file(path, filename='encoder.pickle')
    lb = fetch_file(path, filename='lib.pickle')

    dataframe = pd.DataFrame(columns=["feature", "value", "precision", "recall", "fbeta_score"])
    for feature in cat_features:
        for value in data[feature].unique():
            subset_data = data[data[feature] == value]
            X_test_subset, y_test_subset, encoder, lb = process_data(
                subset_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
            )
            y_pred_subset = model.predict(X_test_subset)
            precision, recall, fbeta = compute_model_metrics(y_test_subset, y_pred_subset)
            dataframe = dataframe.append({"feature": feature, "value":value, "precision": precision, "recall": recall, "fbeta_score": fbeta}, ignore_index = True)
    dataframe.to_csv('slice_output.csv')

if __name__ == "__main__":
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    path = os.path.join(os.getcwd() + "/starter/data")
    data = fetch_file(path, filename="clean_census_data.csv")
    model = train()
    model_slicing(data, cat_features)