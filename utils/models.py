from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    classification_report,
)
from xgboost import XGBClassifier
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_and_predict(
    X_train, X_valid, y_train, y_valid=None, params=None, model_name=None
):

    logger.info(params)
    model = ModelSelector.get_sklearn_model_instance(model_name)(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    # p, r, f1, _ = precision_recall_fscore_support(y_valid, y_pred, labels=[0, 1])
    if y_valid is not None:
        f1 = f1_score(y_valid, y_pred, labels=[0, 1], average="macro")
        classification_report(y_valid, y_pred)
    else:
        f1 = 0.0
    return f1, y_pred, model


class ModelSelector(object):
    _candidates = ["svm", "lr", "mlp", "rf", "xgb", "lgbm"]

    def __init__(self, model_name: str):

        if model_name not in self._candidates:
            raise ValueError(f"must set `model_name` to one of {self._candidates}.")

        self._model_name = model_name

    @classmethod
    def get_sklearn_model_instance(cls, model_name: str) -> BaseEstimator:
        if model_name not in cls._candidates:
            raise ValueError(
                f"model_name must be one of {cls._candidates}.\n",
                f"Now, model_name is `{model_name}`",
            )
        elif model_name == cls._candidates[0]:
            return SVC
        elif model_name == cls._candidates[1]:
            return LogisticRegression
        elif model_name == cls._candidates[2]:
            return MLPClassifier
        elif model_name == cls._candidates[3]:
            return RandomForestClassifier
        elif model_name == cls._candidates[4]:
            return XGBClassifier
        elif model_name == cls._candidates[5]:
            return LGBMClassifier

    def get_model(self):
        if self._model_name == self._candidates[0]:
            return self._get_svm()
        elif self._model_name == self._candidates[1]:
            return self._get_lr()
        elif self._model_name == self._candidates[2]:
            return self._get_mlp()
        elif self._model_name == self._candidates[3]:
            return self._get_rf()
        elif self._model_name == self._candidates[4]:
            return self._get_xgb()
        elif self._model_name == self._candidates[5]:
            return self._get_lightgbm()

    def _get_rf(self):
        model = RandomForestClassifier
        param_grids = {
            "n_estimators": 100,
            "bootstrap": True,
            "criterion": ["gini", "entropy"],
            "max_features": ["auto", "sqrt", "log2"],
            "max_depth": [10, 110],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 10],
        }
        return model, param_grids

    def _get_svm(self):
        model = SVC
        param_grids = {
            "tol": 1e-3,
            "max_iter": 100,
            "C": [10 ** value for value in range(-3, 4)],
            "kernel": ["linear", "rbf"],
            "class_weight": [None, "balanced"],
        }
        return model, param_grids

    def _get_lr(self):
        model = LogisticRegression
        param_grids = {
            "tol": 1e-3,
            "penalty": "l2",
            "max_iter": 10000,
            "dual": False,
            "solver": ["lbfgs", "newton-cg"],
            "C": [10 ** value for value in range(-3, 4)],
            "class_weight": [None, "balanced"],
            "multi_class": ["ovr", "multinomial", "auto"],
        }
        return model, param_grids

    def _get_mlp(self):
        # TODO: 後でpytorchなどDLライブラリで書き直し，embeddingのfinetuneができるようにする
        model = MLPClassifier()
        param_grids = {}
        return model, param_grids

    def _get_xgb(self):
        model = XGBClassifier
        param_grids = {
            "learning_rate": 0.05,
            "n_estimators": 100,
            "objective": "binary:logistic",
            "max_depth": [3, 10],
            "gamma": [1, 10],
            "booster": ["gbtree", "gblinear"],
            "min_child_weight": [1, 6],
            "max_delta_step": [1, 6],
            "subsample": [0.1 * value for value in range(5, 11)],
            "colsample_bytree": [0.1 * value for value in range(5, 11)],
        }
        return model, param_grids

    def _get_lightgbm(self):
        model = LGBMClassifier
        param_grids = {
            "boosting_type": "gbdt",
            "num_leaves": [2, 256],
            "max_depth": -1,
            "learning_rate": [0.005, 0.1],
            "n_estimators": 100,
            "subsample_for_bin": 200000,
            "objective": "binary",
            "class_weight": ["balanced", None],
            "min_split_gain": 0.0,
            "min_child_weight": 0.001,
            "min_child_samples": [5, 100],
            "subsample": [0.4, 1.0],
            "subsample_freq": [1, 7],
            "colsample_bytree": [0.65, 1.0],
            "reg_alpha": [1e-8, 10.0],
            "reg_lambda": [1e-8, 10.0],
            "random_state": 0,
            "n_jobs": 2,
            "silent": True,
            "importance_type": "split",
        }
        return model, param_grids
