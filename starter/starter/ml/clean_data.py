import pandas as pd
import numpy as np

dataset = pd.read_csv("/Users/apple/PycharmProjects/nd0821-c3-starter-code/starter/data/census.csv")
print(dataset)
# dataset = dataset.replace("?", "np.Nan")
df = dataset[(dataset != '?').all(axis=1)]
print(df)