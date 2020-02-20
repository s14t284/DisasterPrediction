import argparse
import json
import logging
import datetime as dt
import pandas as pd
from sklearn.model_selection import KFold
from utils import load_datasets, load_target, train_and_predict
from tqdm import tqdm
from scipy.sparse import coo_matrix


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="./configs/default.json")
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

features = config["features"]
logger.info(features)

target_name = config["target_name"]

logger.info("load datasets")
X_train_all, X_test = load_datasets(features)
y_train_all = load_target(target_name)
logger.info(X_train_all.shape)

fmeasures = []
y_preds = []
models = []

params = config["params"]

kf = KFold(n_splits=10, random_state=RANDOM_STATE)

for train_idx, val_idx in tqdm(kf.split(X_train_all)):
    X_train, X_valid = X_train_all[train_idx, :], X_train_all[val_idx, :]
    y_train, y_valid = y_train_all[train_idx], y_train_all[val_idx]
    f1, y_pred, model = train_and_predict(
        coo_matrix(X_train),
        coo_matrix(X_valid),
        y_train,
        y_valid,
        params,
        config["model_name"],
    )
    fmeasures.append(f1)
    y_preds.append(y_pred)
    models.append(model)

f1score = sum(fmeasures) / len(fmeasures)
print("=== CV scores ===")
print(fmeasures)
print(f1score)
logger.info("=== CV scores ===")
logger.info(fmeasures)
logger.info(f1score)


logger.info("training model for prediction test data...")
_, y_pred, true_model = train_and_predict(
    X_train_all, X_test, y_train_all, params=params, model_name=config["model_name"]
)

logger.info("save predicted result")
sub = pd.DataFrame()
sub[target_name] = y_pred
sub.to_csv("./data/output/sub_{0:%Y%m%d%H%M%S}_{1}.csv".format(now, ""), index=False)
