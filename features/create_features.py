import pandas as pd
import numpy as np
import re
import MeCab
import os
import neologdn

from base import AbstructFeature, get_arguments, generate_features
from healper import (
    in_emoji,
    in_hashtag,
    in_kaomoji,
    in_number,
    in_url,
    RESTRDIC,
    replace_emoji,
    replace_reply,
)


AbstructFeature.DIR = "features/feather"
tagger = MeCab.Tagger("-Owakati")


class Reply(AbstructFeature):
    def create_features(self):
        self._train["Reply"] = train["text"].str.startswith("@") * 1
        self._test["Reply"] = test["text"].str.startswith("@") * 1
        train["text"] = train["text"].apply(replace_reply)
        test["text"] = test["text"].apply(replace_reply)


class URL(AbstructFeature):
    def create_features(self):
        self._train["URL"] = train["text"].apply(in_url) * 1
        self._test["URL"] = test["text"].apply(in_url) * 1
        train["text"] = train["text"].replace(RESTRDIC["url"], "", regex=True)
        test["text"] = test["text"].replace(RESTRDIC["url"], "", regex=True)


class Hashtag(AbstructFeature):
    def create_features(self):
        self._train["Hashtag"] = train["text"].apply(in_hashtag) * 1
        self._test["Hashtag"] = test["text"].apply(in_hashtag) * 1
        train["text"] = train["text"].replace(RESTRDIC["hashtag"], "", regex=True)
        test["text"] = test["text"].replace(RESTRDIC["hashtag"], "", regex=True)


class Kaomoji(AbstructFeature):
    def create_features(self):
        self._train["Kaomoji"] = train["text"].apply(in_kaomoji) * 1
        self._test["Kaomoji"] = test["text"].apply(in_kaomoji) * 1
        train["text"] = train["text"].replace(RESTRDIC["kaomoji"], "", regex=True)
        test["text"] = test["text"].replace(RESTRDIC["kaomoji"], "", regex=True)


class Emoji(AbstructFeature):
    def create_features(self):
        self._train["Emoji"] = train["text"].apply(in_emoji) * 1
        self._test["Emoji"] = test["text"].apply(in_emoji) * 1
        train["text"] = train["text"].apply(replace_emoji)
        test["text"] = test["text"].apply(replace_emoji)


class Number(AbstructFeature):
    def create_features(self):
        self._train["Number"] = train["text"].apply(in_number) * 1
        self._test["Number"] = test["text"].apply(in_number) * 1
        train["text"] = train["text"].replace(RESTRDIC["number"], "0", regex=True)
        test["text"] = test["text"].replace(RESTRDIC["number"], "0", regex=True)


class Mecabbow(AbstructFeature):
    def create_features(self):
        train["text"] = train["text"].apply(neologdn.normalize)
        test["text"] = test["text"].apply(neologdn.normalize)
        self._train["Mecabbow"] = train["text"].apply(tagger.parse)
        self._test["Mecabbow"] = test["text"].apply(tagger.parse)


if __name__ == "__main__":
    args = get_arguments()

    train_path = "./data/input/train.ftr"
    test_path = "./data/input/test.ftr"

    if not os.path.exists(train_path):
        pd.read_csv(args.train).to_feather(train_path)
    if not os.path.exists(test_path):
        pd.read_csv(args.test).to_feather(test_path)

    train = pd.read_feather(train_path)
    test = pd.read_feather(test_path)

    variables = globals()
    generate_features(variables, args.force)
