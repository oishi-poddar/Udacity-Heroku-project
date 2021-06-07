# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics
import pickle
import os
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

    with open("slice_outpt.txt", "w") as file:
        file.write("Precision " + str(precision) +'\n')
        file.write("Recall " + str(recall) + '\n')
        file.write("F-beta score " + str(fbeta))


if __name__ == "__main__":
    train()