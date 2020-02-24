import pandas as pd
import numpy as np
from .models import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


with open("data/slothlib.txt", "r") as f:
    STOP_WORDS = [line.replace("\n", "") for line in f.readlines()]
TFIDF = TfidfVectorizer(
    stop_words=STOP_WORDS, max_features=10000, sublinear_tf=True, ngram_range=(1, 3)
)
CTFIDF = TfidfVectorizer(
    sublinear_tf=True, analyzer="char", ngram_range=(1, 4), max_features=50000,
)
KCV = CountVectorizer()
NKCV = CountVectorizer()


def extract_word_vector(feature, target: str):
    df = pd.read_feather(f"features/feather/{feature}_{target}.ftr")
    return np.apply_along_axis(
        lambda x: np.array([float(v) for v in x[0].split(" ")], dtype="float32"),
        1,
        df.values,
    )


def feature_extractor(features, target: str = "train"):
    dfs = []
    dims = {}
    with tqdm(features) as ftqdm:
        for f in ftqdm:
            ftqdm.set_postfix(processing_feature=f)
            if "bow" in f or "boc" in f or "tfidf" in f or "svd" in f or "topic" in f:
                bow = extract_word_vector(f, target)
                dfs.append(bow)
            elif "nkeyword" in f and "_" not in f:
                keywords = pd.read_feather(f"features/feather/{f}_{target}.ftr")
                if target == "train":
                    NKCV.fit(keywords[f.capitalize()].values)
                kbow = NKCV.transform(keywords[f.capitalize()].values)
                dfs.append(kbow.toarray().astype(np.float32))
            elif "keyword" in f and "_" not in f:
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


def load_datasets(features):
    dfs, dims = feature_extractor(features)
    X_train = np.concatenate(dfs, axis=1)
    dfs, _ = feature_extractor(features, "test")
    X_test = np.concatenate(dfs, axis=1)
    return X_train, X_test, dims


def load_target(target_name):
    train = pd.read_csv("./data/4.true_flg.csv")
    y_train = train[target_name]
    return y_train
