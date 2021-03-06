import argparse
import datetime as dt
import json
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from utils import load_datasets, load_target, train_and_predict


def str_func(x):
    return "bow" if "bow" in x else x


plt.rcParams["font.size"] = 5


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default=["./configs/default.json"])
parser.add_argument("-v", "--voting", default=1, type=int)
options = parser.parse_args()
with open(options.config, "r") as f:
    config = json.load(f)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler1 = logging.StreamHandler()
handler1.setLevel(logging.INFO)
logger.addHandler(handler1)
now = dt.datetime.now()
handler2 = logging.FileHandler(filename="./logs/sub_{0:%Y%m%d%H%M%S}.log".format(now))
logger.addHandler(handler2)
IDNAME = config["ID_name"] if "ID_NAME" in config else "id"
RANDOM_STATE = 0
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

features = config["features"]
logger.info(features)

target_name = config["target_name"]

logger.info("load datasets")
X_train_all, X_test, dims = load_datasets(features)

indexes = [
    f"{str_func(k)}{i}" if v > 1 else str_func(k)
    for k, v in dims.items()
    for i in range(v)
]

y_train_all = load_target(target_name)
logger.info(X_train_all.shape)

fmeasures = []
y_preds = []

params = config["params"]
model_name = config["model_name"]

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

for train_idx, val_idx in tqdm(kf.split(X_train_all, y_train_all)):
    X_train, X_valid = X_train_all[train_idx, :], X_train_all[val_idx, :]
    y_train, y_valid = y_train_all[train_idx], y_train_all[val_idx]
    f1, y_pred, model = train_and_predict(
        csr_matrix(X_train),
        csr_matrix(X_valid),
        y_train,
        y_valid,
        params,
        model_name,
        options.voting,
    )
    fmeasures.append(f1)
    y_preds.append(y_pred)

f1score = sum(fmeasures) / len(fmeasures)
logger.info("=== CV scores ===")
logger.info(fmeasures)
logger.info(f1score)


logger.info("training model for prediction test data...")
_, y_pred, true_model = train_and_predict(
    csr_matrix(X_train_all),
    csr_matrix(X_test),
    y_train_all,
    params=params,
    model_name=model_name,
    voting=options.voting,
)

logger.info("save predicted result")
sub = pd.DataFrame()
sub[target_name] = y_pred
sub.to_csv(
    "./data/output/sub_{0:%Y%m%d%H%M%S}_{1}.csv".format(now, f1score), index=False
)
