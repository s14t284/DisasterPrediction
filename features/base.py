import argparse
import inspect
import re
import time
from abc import ABCMeta, abstractclassmethod
from pathlib import Path
from contextlib import contextmanager

import pandas as pd


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/1.train_data.csv")
    parser.add_argument("--test", type=str, default="data/2.test_data.csv")
    parser.add_argument("-f", "--force", action="store_true")
    return parser.parse_args()


def is_AbstructFeature(v):
    return (
        inspect.isclass(v)
        and issubclass(v, AbstructFeature)
        and not inspect.isabstract(v)
    )


def get_features(namespace: dict):
    for k, v in namespace.items():
        if is_AbstructFeature(v):
            yield v()


def generate_features(namespace: dict, isOverwrite: bool = False):
    for feature in get_features(namespace):
        # feature.run().save()
        if (
            not isOverwrite
            and feature.train_path.exists()
            and feature.test_path.exists()
        ):
            print(f"Generating `{feature.name}` feature was skipped.")
        else:
            feature.run().save()


@contextmanager
def timer(name: str):
    t = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] is done in {time.time() - t:.0f} s")


class AbstructFeature(metaclass=ABCMeta):

    PREFIX = ""
    SUFFIX = ""
    DIR = "."

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub(
                r"([A-Z])", lambda x: f"_{x.group(1).lower()}", self.__class__.__name__
            ).lstrip("_")
        self._train = pd.DataFrame()
        self._test = pd.DataFrame()
        self.train_path = Path(self.DIR) / f"{self.name}_train.ftr"
        self.test_path = Path(self.DIR) / f"{self.name}_test.ftr"

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = f"{self.PREFIX}_" if self.PREFIX else ""
            suffix = f"_{self.SUFFIX}" if self.SUFFIX else ""
            self._train.columns = prefix + self._train.columns + suffix
            self._test.columns = prefix + self._test.columns + suffix
        return self

    @abstractclassmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self._train.to_feather(str(self.train_path))
        self._test.to_feather(str(self.test_path))
