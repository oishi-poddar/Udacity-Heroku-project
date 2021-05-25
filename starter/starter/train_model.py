# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics
import pickle
# Add the necessary imports for the starter code.

# Add code to load in the data.
def train():
    data = pd.read_csv("/Users/apple/PycharmProjects/nd0821-c3-starter-code/starter/data/clean_census_data.csv")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
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
    filename = 'trainedmodel' + '.pkl'
    pickle.dump(model, open("starter/starter" + filename, 'wb'))

    with open('encoder.pickle', 'wb') as f:
        pickle.dump(encoder, f)
    with open('lib.pickle', 'wb') as f:
        pickle.dump(lb, f)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
