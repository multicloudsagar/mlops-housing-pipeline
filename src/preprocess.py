# src/preprocess.py
from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_data():
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    df.to_csv("data/raw/california_housing.csv", index=False)

if __name__ == "__main__":
    load_data()
