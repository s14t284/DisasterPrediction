import argparse
import datetime as dt
import json
import logging
import random
from distutils.util import strtobool

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold

from utils import ModelSelector, load_datasets, load_target, train_and_predict


def str_func(x):
    return "bow" if "bow" in x else x


plt.rcParams["font.size"] = 5


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="./configs/default.json")
parser.add_argument("-n", "--n_trials", default=100, type=int)
options = parser.parse_args()
with open(options.config, "r") as f:
    config = json.load(f)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler1 = logging.StreamHandler()
handler1.setLevel(logging.INFO)
logger.addHandler(handler1)
now = dt.datetime.now()
handler2 = logging.FileHandler(filename="./logs/tune_{0:%Y%m%d%H%M%S}.log".format(now))
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


def param_grids_to_params(trial: optuna.Trial, param_grids: dict):
    params = {}
    for k, v in param_grids.items():
        # set optimizing target parameters
        if isinstance(v, list):
            if len(v) > 2:
                params[k] = trial.suggest_categorical(k, v)
            elif all([isinstance(s, bool) for s in v]):
                b = strtobool(trial.suggest_categorical(k, [str(p) for p in v]))
                params[k] = True if b == 1 else False
            elif type(v[0]) == int:
                params[k] = trial.suggest_int(k, v[0], v[1])
            elif type(v[0]) == float:
                params[k] = trial.suggest_uniform(k, v[0], v[1])
            else:
                params[k] = trial.suggest_categorical(k, v)
        # set static parameters
        else:
            params[k] = v
    return params


def objective(trial: optuna.Trial):
    fmeasures = []
    ms = ModelSelector(config["model_name"])
    _, param_grids = ms.get_model()
    params = param_grids_to_params(trial, param_grids)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, val_idx in kf.split(X_train_all, y_train_all):
        X_train, X_valid = X_train_all[train_idx, :], X_train_all[val_idx, :]
        y_train, y_valid = y_train_all[train_idx], y_train_all[val_idx]
        f1, _, _ = train_and_predict(
            csr_matrix(X_train),
            csr_matrix(X_valid),
            y_train,
            y_valid,
            params,
            config["model_name"],
        )
        fmeasures.append(f1)

    f1score = sum(fmeasures) / len(fmeasures)
    return f1score


study = optuna.create_study(direction="maximize")
logger.info("tuning model...")
study.optimize(objective, n_trials=options.n_trials)
params = study.best_trial.params
logger.info(f"number of finished trials: {str(len(study.trials))}")
logger.info(f"best trial: {str(params)}")
best_score = study.best_value
logger.info(f"best score: {str(best_score)}")
json_name = "tuned_{0:%Y%m%d%H%M%S}_{1}.json".format(now, config["model_name"])
logger.info(f"save best params to `configs/{json_name}`")
with open(f"./configs/{json_name}", "w") as f:
    for k, v in params.items():
        config["params"][k] = v
    json.dump(config, f)

_, y_pred, true_model = train_and_predict(
    X_train_all, X_test, y_train_all, params=params, model_name=config["model_name"]
)

logger.info("save predicted result")
sub = pd.DataFrame()
sub[target_name] = y_pred
sub.to_csv(
    "./data/output/tune_{0:%Y%m%d%H%M%S}_{1}.csv".format(now, best_score), index=False
)

if config["model_name"] in ["rf", "xgb", "lgbm"]:
    importance = pd.DataFrame(
        true_model.feature_importances_, index=indexes, columns=["importance"]
    )
    importance = importance.sort_values("importance", ascending=False)
    importance.head(50).plot.bar()
    plt.savefig("logs/tune_{0:%Y%m%d%H%M%S}_feature_importance.png".format(now))
    plt.close()
    logger.info(importance)
