import pandas as pd
import numpy as np
from .models import *
from sklearn.feature_extraction.text import CountVectorizer


with open("data/slothlib.txt", "r") as f:
    STOP_WORDS = [line.replace("\n", "") for line in f.readlines()]
CV = CountVectorizer(stop_words=STOP_WORDS)


def feature_extractor(features, target: str = "train"):
    dfs = []
    for f in features:
        if "bow" in f:
            texts = pd.read_feather(f"features/feather/{f}_{target}.ftr")
            if target == "train":
                CV.fit(texts[f.capitalize()].values)
            bow = CV.transform(texts[f.capitalize()].values)
            dfs.append(bow.toarray().astype(np.float32))
        else:
            dfs.append(pd.read_feather(f"features/feather/{f}_{target}.ftr").to_numpy().astype(np.float32))
    return dfs


def load_datasets(features):
    dfs = feature_extractor(features)
    X_train = np.concatenate(dfs, axis=1)
    dfs = feature_extractor(features, "test")
    X_test = np.concatenate(dfs, axis=1)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv("./data/1.train_data.csv")
    y_train = train[target_name]
    return y_train
