import pandas as pd
import numpy as np
from .models import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


with open("data/slothlib.txt", "r") as f:
    STOP_WORDS = [line.replace("\n", "") for line in f.readlines()]
TFIDF = TfidfVectorizer(stop_words=STOP_WORDS, max_features=10000, sublinear_tf=True,)
CTFIDF = TfidfVectorizer(
    sublinear_tf=True, analyzer="char", ngram_range=(1, 4), max_features=50000,
)
KCV = CountVectorizer()
SVD = TruncatedSVD(1000)
SCL = StandardScaler()


def extract_word_vector(feature, target: str, dim_reduc: bool = False):
    texts = pd.read_feather(f"features/feather/{feature}_{target}.ftr")
    corpus = texts[feature.capitalize()].values
    if target == "train":
        TFIDF.fit(corpus)
        CTFIDF.fit(corpus)
    bow = TFIDF.transform(corpus).toarray().astype(np.float32)
    boc = CTFIDF.transform(corpus).toarray().astype(np.float32)
    bow = np.hstack((bow, boc))
    if dim_reduc:
        if target == "train":
            SVD.fit(bow)
        bow_svd = SVD.transform(bow)
        if target == "train":
            SCL.fit(bow_svd)
        bow_scl = SCL.transform(bow_svd)
        return bow_scl
    return bow


def feature_extractor(features, target: str = "train", dim_reduc: bool = False):
    dfs = []
    dims = {}
    for f in features:
        if "bow" in f:
            bow = extract_word_vector(f, target, dim_reduc)
            dfs.append(bow)
        elif "keyword" in f:
            keywords = pd.read_feather(f"features/feather/{f}_{target}.ftr")
            if target == "train":
                KCV.fit(keywords[f.capitalize()].values)
            kbow = KCV.transform(keywords[f.capitalize()].values)
            dfs.append(kbow.toarray().astype(np.float32))
        else:
            dfs.append(
                pd.read_feather(f"features/feather/{f}_{target}.ftr")
                .to_numpy()
                .astype(np.float32)
            )
        dims[f] = dfs[-1].shape[1]
    return dfs, dims


def load_datasets(features, dim_reduc: bool):
    dfs, dims = feature_extractor(features, dim_reduc=dim_reduc)
    X_train = np.concatenate(dfs, axis=1)
    dfs, _ = feature_extractor(features, "test", dim_reduc=dim_reduc)
    X_test = np.concatenate(dfs, axis=1)
    return X_train, X_test, dims


def load_target(target_name):
    train = pd.read_csv("./data/4.true_flg.csv")
    y_train = train[target_name]
    return y_train
