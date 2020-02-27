import argparse
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
import json

from utils import load_datasets, load_target
from utils.models import ModelSelector


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="./configs/default.json")
options = parser.parse_args()
with open(options.config, "r") as f:
    config = json.load(f)
features = config["features"]
target_name = config["target_name"]
X_train_all, X_test, dims = load_datasets(features)
y_train_all = load_target(target_name)
indexes = [k for k, v in dims.items() for i in range(v)]

clf = ModelSelector.get_sklearn_model_instance(config["model_name"])(**config["params"])
sfm = SelectFromModel(clf)
print("fitting for selecting features...")
sfm.fit(X_train_all, y_train_all)
n_features = sfm.transform(X_train_all).shape[1]

config["features"] = sorted(
    list(set([k for k, v in zip(indexes, sfm.get_support()) if v]))
)
print(config["features"])
print(
    "original features count: {}, selected features count: {}".format(
        X_train_all.shape[1], n_features
    )
)

with open(options.config.replace(".json", "_selected_feature.json"), "w") as f:
    json.dump(config, f)
