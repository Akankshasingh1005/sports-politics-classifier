import pandas as pd

def load_data(path="bbc-text.csv"):
    
    df = pd.read_csv(path)

    #keep only sport and politics
    df = df[df["category"].isin(["sport", "politics"])]

    df["label"] = df["category"].map({
        "sport": 0,
        "politics": 1
    })

    X = df["text"].values
    y = df["label"].values

    return X, y
