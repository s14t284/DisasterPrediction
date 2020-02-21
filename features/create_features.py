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
    in_reply,
    count_date_representation,
    RESTRDIC,
    replace_emoji,
    replace_reply,
    load_csv_for_train_to_df,
)


AbstructFeature.DIR = "features/feather"
with open("./data/slothlib.txt") as f:
    STOPWORDS = [line.replace("\n", "") for line in f.readlines()]
tagger = MeCab.Tagger("")


class Keyword(AbstructFeature):
    def create_features(self):
        self._train["Keyword"] = train["keyword"]
        self._test["Keyword"] = test["keyword"]


class Reply(AbstructFeature):
    def create_features(self):
        self._train["Reply"] = train["text"].apply(in_reply) * 1
        self._test["Reply"] = test["text"].apply(in_reply) * 1
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


class Length(AbstructFeature):
    def create_features(self):
        self._train["Length"] = train["text"].apply(len)
        self._test["Length"] = test["text"].apply(len)


class Punctuation(AbstructFeature):

    PUNCS = ["、", "，", ",", "。", "．", "."]

    def create_features(self):
        self._train["Length"] = train["text"].apply(self.count_punc)
        self._test["Length"] = test["text"].apply(self.count_punc)

    @classmethod
    def count_punc(cls, text):
        return len([c for c in text if c in cls.PUNCS])


class DateRepresentation(AbstructFeature):
    def create_features(self):
        self._train["Length"] = train["text"].apply(count_date_representation)
        self._test["Length"] = test["text"].apply(count_date_representation)


class Mecabbow(AbstructFeature):
    def create_features(self):
        train["text"] = train["text"].apply(self.normalize)
        test["text"] = test["text"].apply(self.normalize)
        self._train["Mecabbow"] = train["text"].apply(self.parse)
        self._test["Mecabbow"] = test["text"].apply(self.parse)

    @classmethod
    def parse(cls, text):
        node = tagger.parseToNode(text)
        node = node.next
        words = []
        while node.next:
            if node.feature.split(",")[0] in ["名詞", "動詞", "形容詞", "副詞"]:
                words.append(node.surface)
            node = node.next
        return " ".join(words)

    @classmethod
    def normalize(cls, text):
        text = re.sub(r"\u3000", " ", text)
        new_text = neologdn.normalize(text, repeat=1)
        new_text = re.sub(r"[【】『』「」\[\]\(\)（）［］\|｜]", " ", new_text)
        new_text = new_text.lower()
        return new_text


class Mwordcount(AbstructFeature):
    """mecabで形態素解析したときの単語数

    Args:
        AbstructFeature ([type]): [description]

    Returns:
        [type]: [description]
    """

    tagger = MeCab.Tagger("-Owakati")

    def create_features(self):
        self._train["Mwordcount"] = train["text"].apply(self.word_count)
        self._test["Mwordcount"] = test["text"].apply(self.word_count)

    @classmethod
    def word_count(cls, text) -> int:
        return len(cls.parse(text).split(" ")) + 1

    @classmethod
    def parse(cls, text):
        return tagger.parse(text).replace(" \n", "")


class Muniquecount(AbstructFeature):
    """mecabで形態素解析したときの文に含まれる単語の種類

    Args:
        AbstructFeature ([type]): [description]

    Returns:
        [type]: [description]
    """

    tagger = MeCab.Tagger("-Owakati")

    def create_features(self):
        self._train["Muniquecount"] = train["text"].apply(self.word_count)
        self._test["Muniquecount"] = test["text"].apply(self.word_count)

    @classmethod
    def word_count(cls, text) -> int:
        return len(list(set(cls.parse(text).split(" ")))) + 1

    @classmethod
    def parse(cls, text):
        return tagger.parse(text).replace(" \n", "")


class Mstopcount(AbstructFeature):
    """mecabで形態素解析したときの文に含まれるstopwordの数
    stopwordにはslothlibを使用

    Args:
        AbstructFeature ([type]): [description]

    Returns:
        [type]: [description]
    """

    tagger = MeCab.Tagger("-Owakati")

    def create_features(self):
        self._train["Mstopcount"] = train["text"].apply(self.word_count)
        self._test["Mstopcount"] = test["text"].apply(self.word_count)

    @classmethod
    def word_count(cls, text) -> int:
        return len([w for w in cls.parse(text).split(" ") if w in STOPWORDS])

    @classmethod
    def parse(cls, text):
        return tagger.parse(text).replace(" \n", "")


class Mmeanlen(AbstructFeature):
    """mecabで形態素解析したときの文中の単語の平均長

    Args:
        AbstructFeature ([type]): [description]

    Returns:
        [type]: [description]
    """

    tagger = MeCab.Tagger("-Owakati")

    def create_features(self):
        self._train["Mstopcount"] = train["text"].apply(self.word_count)
        self._test["Mstopcount"] = test["text"].apply(self.word_count)

    @classmethod
    def word_count(cls, text) -> int:
        word_lens = [len(w) for w in cls.parse(text).split(" ")]
        return sum(word_lens) / len(word_lens) if len(word_lens) > 0 else 0

    @classmethod
    def parse(cls, text):
        return tagger.parse(text).replace(" \n", "")


if __name__ == "__main__":
    args = get_arguments()

    train_path = "./data/input/train.ftr"
    test_path = "./data/input/test.ftr"

    if not os.path.exists(train_path):
        load_csv_for_train_to_df(args.train).to_feather(train_path)
    if not os.path.exists(test_path):
        pd.read_csv(args.test).to_feather(test_path)

    train = pd.read_feather(train_path)
    test = pd.read_feather(test_path)

    variables = globals()
    generate_features(variables, args.force)
