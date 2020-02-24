import inspect
import os
import re
import string
import unicodedata
from collections import defaultdict
from typing import List, Union

import MeCab
import neologdn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation as LDA, TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler

from base import AbstructFeature, generate_features, get_arguments, is_AbstructFeature
from healper import (
    RESTRDIC,
    count_capital_name,
    count_date_representation,
    count_emoji,
    count_hashtag,
    count_japan_clues,
    count_kaomoji,
    count_number,
    count_place_clues,
    count_prefecture_name,
    count_region_name,
    count_reply,
    count_roma_capital_name,
    count_roma_prefecture_name,
    count_url,
    in_eng,
    load_csv_for_train_to_df,
    replace_emoji,
    replace_reply,
)

AbstructFeature.DIR = "features/feather"
with open("./data/slothlib.txt") as f:
    STOPWORDS = [line.replace("\n", "") for line in f.readlines()]


tagger = MeCab.Tagger("")
TOPIC_KEYWORDS = {}
RANDOM_STATE = 0


def vec_to_vecstrs(vec: list):
    """
    [[v1, v2, ..., vn], [v1, ...]] -> ["v1 v2 ... vn", "v1 ..."]
    Args:
        bow (list): [description]

    Returns:
        [type]: [description]
    """

    vec_strs = []
    for b in vec:
        vec_strs.append(" ".join([str(i) for i in b]))
    return vec_strs


def get_num_from_rank(i: int, rank: Union[int, List[int]]):
    if isinstance(rank, int):
        return str(i)[rank]
    elif isinstance(rank, list):
        if len(rank) > 1:
            return str(i)[rank[0] : rank[1] + 1]
        else:
            return str(i)[rank[0]]


class Keyword(AbstructFeature):
    def create_features(self):
        CV = CountVectorizer()
        train_keywords = train["keyword"].values.tolist()
        test_keywords = test["keyword"].values.tolist()
        corpus = train_keywords + test_keywords
        CV.fit(corpus)
        train_kbow = (
            CV.transform(train_keywords).toarray().astype(np.float32)
        )
        train_kbow_list = train_kbow.tolist()
        self._train["Keyword"] = vec_to_vecstrs(train_kbow_list)
        test_kbow = CV.transform(test_keywords).toarray().astype(np.float32)
        test_kbow_list = test_kbow.tolist()
        self._test["Keyword"] = vec_to_vecstrs(test_kbow_list)


class QueryKeywordNum(AbstructFeature):
    """
    queryに用いて引っかかったkeywordの数
    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Query_keyword_num"] = train["keyword"].apply(
            lambda x: len(x.split(" "))
        )
        self._test["Query_keyword_num"] = test["keyword"].apply(
            lambda x: len(x.split(" "))
        )


class CountKeyword(AbstructFeature):
    def create_features(self):
        count_keyword_dict = train["keyword"].value_counts().to_dict()
        self._train["Count_keyword"] = train["keyword"].apply(
            lambda x: count_keyword_dict[x] if x in count_keyword_dict else 0
        )
        self._test["Count_keyword"] = test["keyword"].apply(
            lambda x: count_keyword_dict[x] if x in count_keyword_dict else 0
        )


class RatioAnsKeyword(AbstructFeature):
    """keywordが出現するときの正解割合

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_ans_ratio = self.calc_count_ans_ratio(train)
        self._train["Ratio_ans_keyword"] = train["keyword"].apply(
            lambda x: count_ans_ratio[x] if x in count_ans_ratio else 0
        )
        self._test["Ratio_ans_keyword"] = test["keyword"].apply(
            lambda x: count_ans_ratio[x] if x in count_ans_ratio else 0
        )

    @classmethod
    def calc_count_ans_ratio(cls, df):
        keywords = df["keyword"].to_list()
        count_keyword_with_flg = df.groupby("keyword").flg.value_counts().to_dict()
        key_freq_store = {k: defaultdict(int) for k in keywords}
        for k, v in count_keyword_with_flg.items():
            key_freq_store[k[0]][k[1]] = v
        key_ratio_store = {}
        for k in keywords:
            if 0 in key_freq_store[k] and 1 in key_freq_store[k]:
                key_ratio_store[k] = key_freq_store[k][1] / (
                    key_freq_store[k][0] + key_freq_store[k][1]
                )
            else:
                key_ratio_store[k] = 0.0
        return key_ratio_store


class Nkeyword(AbstructFeature):
    def create_features(self):
        train["nkeyword"] = train["keyword"].apply(self.replace_rule)
        test["nkeyword"] = test["keyword"].apply(self.replace_rule)
        global TOPIC_KEYWORDS
        TOPIC_KEYWORDS = (
            train.nkeyword[train.nkeyword.apply(lambda x: len(x.split(" "))) == 1]
            .value_counts()
            .to_dict()
        )

        CV = CountVectorizer()
        train_keywords = train["nkeyword"].values.tolist()
        test_keywords = test["nkeyword"].values.tolist()
        corpus = train_keywords + test_keywords
        CV.fit(corpus)
        train_kbow = CV.transform(train_keywords).toarray().astype(np.float32)
        train_kbow_list = train_kbow.tolist()
        self._train["Nkeyword"] = vec_to_vecstrs(train_kbow_list)
        test_kbow = CV.transform(test_keywords).toarray().astype(np.float32)
        test_kbow_list = test_kbow.tolist()
        self._test["Nkeyword"] = vec_to_vecstrs(test_kbow_list)

        train["topic_keyword"] = train.nkeyword.apply(self.decide_keyword)
        test["topic_keyword"] = test.nkeyword.apply(self.decide_keyword)

    @classmethod
    def replace_rule(cls, text: str):
        words = text.split(" ")
        for i in range(len(words)):
            words[i] = re.sub(r"(する|した|の|犯|者|官|事態|証言)$", "", words[i])
            words[i] = re.sub(r"\S+テロ", "テロ", words[i])
            words[i] = re.sub(r"\S+雷", "雷", words[i])
            words[i] = re.sub(r"\S+衝突", "衝突", words[i])
            words[i] = re.sub(r"\S*殺人\S*", "殺人", words[i])
        words = sorted(list(set(words)))
        return " ".join(words)

    @classmethod
    def decide_keyword(cls, keywords: str):
        keyword_list = keywords.split(" ")
        decided = keyword_list[0]
        max_val = TOPIC_KEYWORDS[decided]
        for k in keyword_list[1:]:
            if max_val < TOPIC_KEYWORDS[k]:
                decided = k
                max_val = TOPIC_KEYWORDS[k]
        return decided


class QueryNkeywordNum(AbstructFeature):
    """
    queryに用いて引っかかった整形後のkeywordの数
    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Query_nkeyword_num"] = train["nkeyword"].apply(
            lambda x: len(x.split(" "))
        )
        self._test["Query_nkeyword_num"] = test["nkeyword"].apply(
            lambda x: len(x.split(" "))
        )


class CountNkeyword(AbstructFeature):
    """整形後のkeywordの学習データ中での出現数

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_keyword_dict = train["nkeyword"].value_counts().to_dict()
        self._train["Count_keyword"] = train["nkeyword"].apply(
            lambda x: count_keyword_dict[x] if x in count_keyword_dict else 0
        )
        self._test["Count_keyword"] = test["nkeyword"].apply(
            lambda x: count_keyword_dict[x] if x in count_keyword_dict else 0
        )


class RatioAnsNkeyword(AbstructFeature):
    """整形後のkeywordが出現するときの正解割合

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_ans_ratio = self.calc_count_ans_ratio(train)
        self._train["Ratio_ans_nkeyword"] = train["nkeyword"].apply(
            lambda x: count_ans_ratio[x] if x in count_ans_ratio else 0
        )
        self._test["Ratio_ans_nkeyword"] = test["nkeyword"].apply(
            lambda x: count_ans_ratio[x] if x in count_ans_ratio else 0
        )

    @classmethod
    def calc_count_ans_ratio(cls, df):
        keywords = df["nkeyword"].to_list()
        count_keyword_with_flg = df.groupby("nkeyword").flg.value_counts().to_dict()
        key_freq_store = {k: defaultdict(int) for k in keywords}
        for k, v in count_keyword_with_flg.items():
            key_freq_store[k[0]][k[1]] = v
        key_ratio_store = {}
        for k in keywords:
            if 0 in key_freq_store[k] and 1 in key_freq_store[k]:
                key_ratio_store[k] = key_freq_store[k][1] / (
                    key_freq_store[k][0] + key_freq_store[k][1]
                )
            else:
                key_ratio_store[k] = 0.0
        return key_ratio_store


class Prefecture(AbstructFeature):
    def create_features(self):
        train["location"] = train.fillna({"location": "nan"})["location"]
        test["location"] = test.fillna({"location": "nan"})["location"]
        self._train["Prefecture"] = train["location"].apply(count_prefecture_name)
        self._test["Prefecture"] = test["location"].apply(count_prefecture_name)


class Region(AbstructFeature):
    def create_features(self):
        self._train["Region"] = train["location"].apply(count_region_name)
        self._test["Region"] = test["location"].apply(count_region_name)


class Capital(AbstructFeature):
    def create_features(self):
        self._train["Capital"] = train["location"].apply(count_capital_name)
        self._test["Capital"] = test["location"].apply(count_capital_name)


class Lineng(AbstructFeature):
    def create_features(self):
        self._train["Lineng"] = train["location"].apply(in_eng)
        self._test["Lineng"] = test["location"].apply(in_eng)


class Rprefecture(AbstructFeature):
    def create_features(self):
        self._train["Rprefecture"] = train["location"].apply(count_roma_prefecture_name)
        self._test["Rprefecture"] = test["location"].apply(count_roma_prefecture_name)


class Rcapital(AbstructFeature):
    def create_features(self):
        self._train["Rcapital"] = train["location"].apply(count_roma_capital_name)
        self._test["Rcapital"] = test["location"].apply(count_roma_capital_name)


class Placeclues(AbstructFeature):
    def create_features(self):
        self._train["Placeclues"] = train["location"].apply(count_place_clues)
        self._test["Placeclues"] = test["location"].apply(count_place_clues)


class Japanclues(AbstructFeature):
    def create_features(self):
        self._train["Japanclues"] = train["location"].apply(count_japan_clues)
        self._test["Japanclues"] = test["location"].apply(count_japan_clues)


class Id(AbstructFeature):
    def create_features(self):
        self._train["Id"] = train["id"].apply(lambda x: int(x))
        self._test["Id"] = test["id"].apply(lambda x: int(x))


class NormalizeId(AbstructFeature):
    """
    leaked feature
    normalize id value using train
    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        normalize_term = train.id.max()
        self._train["Normalize_id"] = train.id / normalize_term
        self._test["Normalize_id"] = test.id / normalize_term


class Uniqid(AbstructFeature):
    """id中のuniqueな数字の割合

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Uniqid"] = train["id"].apply(
            lambda x: len([s for s in str(x)]) / len(str(x))
        )
        self._test["Uniqid"] = test["id"].apply(
            lambda x: len([s for s in str(x)]) / len(str(x))
        )


class Id1(AbstructFeature):
    """idの1の位

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Id1"] = train["id"].apply(lambda x: get_num_from_rank(x, 4))
        self._test["Id1"] = test["id"].apply(lambda x: get_num_from_rank(x, 4))
        train["Id1"] = self._train["Id1"]
        test["Id1"] = self._test["Id1"]


class Id10(AbstructFeature):
    """idの10の位

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Id10"] = train["id"].apply(lambda x: get_num_from_rank(x, 3))
        self._test["Id10"] = test["id"].apply(lambda x: get_num_from_rank(x, 3))
        train["Id10"] = self._train["Id10"]
        test["Id10"] = self._test["Id10"]


class Id100(AbstructFeature):
    """idの100の位

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Id100"] = train["id"].apply(lambda x: get_num_from_rank(x, 2))
        self._test["Id100"] = test["id"].apply(lambda x: get_num_from_rank(x, 2))
        train["Id100"] = self._train["Id100"]
        test["Id100"] = self._test["Id100"]


class Id1000(AbstructFeature):
    """idの1000の位

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Id1000"] = train["id"].apply(lambda x: get_num_from_rank(x, 1))
        self._test["Id1000"] = test["id"].apply(lambda x: get_num_from_rank(x, 1))
        train["Id1000"] = self._train["Id1000"]
        test["Id1000"] = self._test["Id1000"]


class Id10000(AbstructFeature):
    """idの10000の位

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Id10000"] = train["id"].apply(lambda x: get_num_from_rank(x, 0))
        self._test["Id10000"] = test["id"].apply(lambda x: get_num_from_rank(x, 0))
        train["Id10000"] = self._train["Id10000"]
        test["Id10000"] = self._test["Id10000"]


class Id10_1(AbstructFeature):
    """idの10の位と1の位

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Id10_1"] = train["id"].apply(
            lambda x: get_num_from_rank(x, [3, 4])
        )
        self._test["Id10_1"] = test["id"].apply(lambda x: get_num_from_rank(x, [3, 4]))
        train["Id10_1"] = self._train["Id10_1"]
        test["Id10_1"] = self._test["Id10_1"]


class Id100_10(AbstructFeature):
    """idの100の位と10の位

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Id100_10"] = train["id"].apply(
            lambda x: get_num_from_rank(x, [2, 3])
        )
        self._test["Id100_10"] = test["id"].apply(
            lambda x: get_num_from_rank(x, [2, 3])
        )
        train["Id100_10"] = self._train["Id100_10"]
        test["Id100_10"] = self._test["Id100_10"]


class Id1000_100(AbstructFeature):
    """idの1000の位と100の位

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Id1000_100"] = train["id"].apply(
            lambda x: get_num_from_rank(x, [1, 2])
        )
        self._test["Id1000_100"] = test["id"].apply(
            lambda x: get_num_from_rank(x, [1, 2])
        )
        train["Id1000_100"] = self._train["Id1000_100"]
        test["Id1000_100"] = self._test["Id1000_100"]


class Id10000_1000(AbstructFeature):
    """idの10000の位と1000の位

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        self._train["Id10000_1000"] = train["id"].apply(
            lambda x: get_num_from_rank(x, [0, 1])
        )
        self._test["Id10000_1000"] = test["id"].apply(
            lambda x: get_num_from_rank(x, [0, 1])
        )
        train["Id10000_1000"] = self._train["Id10000_1000"]
        test["Id10000_1000"] = self._test["Id10000_1000"]


class CountId1(AbstructFeature):
    """idの1の位の出現数

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_id_dict = train["Id1"].value_counts().to_dict()
        self._train["Count_id1"] = train["Id1"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )
        self._test["Count_id1"] = test["Id1"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )


class CountId10(AbstructFeature):
    """idの10の位の出現数

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_id_dict = train["Id10"].value_counts().to_dict()
        self._train["Count_id10"] = train["Id10"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )
        self._test["Count_id10"] = test["Id10"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )


class CountId100(AbstructFeature):
    """idの100の位の出現数

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_id_dict = train["Id100"].value_counts().to_dict()
        self._train["Count_id100"] = train["Id100"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )
        self._test["Count_id100"] = test["Id100"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )


class CountId1000(AbstructFeature):
    """idの1000の位の出現数

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_id_dict = train["Id1000"].value_counts().to_dict()
        self._train["Count_id1000"] = train["Id1000"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )
        self._test["Count_id1000"] = test["Id1000"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )


class CountId10000(AbstructFeature):
    """idの10000の位

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_id_dict = train["Id10000"].value_counts().to_dict()
        self._train["Count_id10000"] = train["Id10000"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )
        self._test["Count_id10000"] = test["Id10000"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )


class CountId10_1(AbstructFeature):
    """idの10の位と1の位の出現数

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_id_dict = train["Id10_1"].value_counts().to_dict()
        self._train["Count_id10_1"] = train["Id10_1"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )
        self._test["Count_id10_1"] = test["Id10_1"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )


class CountId100_10(AbstructFeature):
    """idの100の位と10の位の出現数

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_id_dict = train["Id100_10"].value_counts().to_dict()
        self._train["Count_id100_10"] = train["Id100_10"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )
        self._test["Count_id100_10"] = test["Id100_10"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )


class CountId1000_100(AbstructFeature):
    """idの1000の位と100の位の出現数

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_id_dict = train["Id1000_100"].value_counts().to_dict()
        self._train["Count_id1000_100"] = train["Id1000_100"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )
        self._test["Count_id1000_100"] = test["Id1000_100"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )


class CountId10000_1000(AbstructFeature):
    """idの10000の位と1000の位の出現数

    Args:
        AbstructFeature ([type]): [description]
    """

    def create_features(self):
        count_id_dict = train["Id10000_1000"].value_counts().to_dict()
        self._train["Count_id10000_1000"] = train["Id10000_1000"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )
        self._test["Count_id10000_1000"] = test["Id10000_1000"].apply(
            lambda x: count_id_dict[x] if x in count_id_dict else 0
        )


class Reply(AbstructFeature):
    def create_features(self):
        self._train["Reply"] = train["text"].apply(count_reply)
        self._test["Reply"] = test["text"].apply(count_reply)
        train["text"] = train["text"].apply(replace_reply)
        test["text"] = test["text"].apply(replace_reply)


class URL(AbstructFeature):
    def create_features(self):
        self._train["URL"] = train["text"].apply(count_url)
        self._test["URL"] = test["text"].apply(count_url)
        train["text"] = train["text"].replace(RESTRDIC["url"], "", regex=True)
        test["text"] = test["text"].replace(RESTRDIC["url"], "", regex=True)


class Hashtag(AbstructFeature):
    def create_features(self):
        self._train["Hashtag"] = train["text"].apply(count_hashtag)
        self._test["Hashtag"] = test["text"].apply(count_hashtag)
        train["text"] = train["text"].replace(RESTRDIC["hashtag"], "", regex=True)
        test["text"] = test["text"].replace(RESTRDIC["hashtag"], "", regex=True)


class Kaomoji(AbstructFeature):
    def create_features(self):
        self._train["Kaomoji"] = train["text"].apply(count_kaomoji)
        self._test["Kaomoji"] = test["text"].apply(count_kaomoji)
        train["text"] = train["text"].replace(RESTRDIC["kaomoji"], "", regex=True)
        test["text"] = test["text"].replace(RESTRDIC["kaomoji"], "", regex=True)


class Emoji(AbstructFeature):
    def create_features(self):
        self._train["Emoji"] = train["text"].apply(count_emoji)
        self._test["Emoji"] = test["text"].apply(count_emoji)
        train["text"] = train["text"].apply(replace_emoji)
        test["text"] = test["text"].apply(replace_emoji)


class Number(AbstructFeature):
    def create_features(self):
        self._train["Number"] = train["text"].apply(count_number)
        self._test["Number"] = test["text"].apply(count_number)
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


class Countexclamation(AbstructFeature):
    def create_features(self):
        self._train["Countexclamation"] = train["text"].apply(self.count_exclamation)
        self._test["Countexclamation"] = test["text"].apply(self.count_exclamation)

    @classmethod
    def count_exclamation(cls, text: str):
        text = neologdn.normalize(text)
        return len(re.findall(r"!", text))


class Countquestion(AbstructFeature):
    def create_features(self):
        self._train["Countquestion"] = train["text"].apply(self.count_question)
        self._test["Countquestion"] = test["text"].apply(self.count_question)

    @classmethod
    def count_question(cls, text: str):
        text = neologdn.normalize(text)
        return len(re.findall(r"\?", text))


class Countlaugh(AbstructFeature):
    def create_features(self):
        self._train["Countlaugh"] = train["text"].apply(self.count_laugh)
        self._test["Countlaugh"] = test["text"].apply(self.count_laugh)

    @classmethod
    def count_laugh(cls, text: str):
        text = neologdn.normalize(text)
        return len(re.findall(r"\Ww", text))


class Mecabbow(AbstructFeature):
    def create_features(self):
        train["text"] = train["text"].apply(self.normalize)
        test["text"] = test["text"].apply(self.normalize)
        train["mecab_text"] = train["text"].apply(self.parse)
        test["mecab_text"] = test["text"].apply(self.parse)
        global corpus_bow
        corpus_bow = np.concatenate([train["mecab_text"].values, test["mecab_text"].values])
        CV = CountVectorizer(stop_words=STOPWORDS, max_features=10000)
        CV.fit(corpus_bow)
        global train_bow
        train_bow = (
            CV.transform(train["mecab_text"].values).toarray().astype(np.float32)
        )
        train_bow_list = train_bow.tolist()
        self._train["Mecabbow"] = vec_to_vecstrs(train_bow_list)
        global test_bow
        test_bow = CV.transform(test["mecab_text"].values).toarray().astype(np.float32)
        test_bow_list = test_bow.tolist()
        self._test["Mecabbow"] = vec_to_vecstrs(test_bow_list)

    @classmethod
    def parse(cls, text):
        node = tagger.parseToNode(text)
        node = node.next
        words = []
        while node.next:
            if (
                True
                # node.feature.split(",")[0] in ["名詞", "動詞", "形容詞", "副詞"]
                # and len(node.surface) > 1
                # and node.surface.lower() not in cls.ASCII_LOWERCASES
            ):
                words.append(node.surface)
            node = node.next
        return " ".join(words)

    @classmethod
    def normalize(cls, text):
        text = re.sub(r"\u3000", " ", text)
        new_text = neologdn.normalize(text, repeat=1)
        new_text = re.sub(r"[【】『』「」\[\]\(\)（）［］\|｜<>＜＞《》\"\\\/!\?]+", " ", new_text)
        new_text = unicodedata.normalize("NFKC", new_text.lower())
        table = str.maketrans("", "", string.punctuation + "「」、。・※")
        new_text = new_text.translate(table)
        return new_text


class Mecabwordtfidf(AbstructFeature):
    def create_features(self):
        global train_bow
        global test_bow
        TFIDF = TfidfTransformer(sublinear_tf=True)
        TFIDF.fit(np.concatenate([train_bow, test_bow]))
        train_tfidf = TFIDF.transform(train_bow).toarray().tolist()
        test_tfidf = TFIDF.transform(test_bow).toarray().tolist()
        self._train["Mecabwordtfidf"] = vec_to_vecstrs(train_tfidf)
        self._test["Mecabwordtfidf"] = vec_to_vecstrs(test_tfidf)


class Mecabwordsvd100(AbstructFeature):
    DIMNUM = 100
    key = f"Mecabwordsvd{DIMNUM}"

    def create_features(self):
        global train_bow
        global test_bow
        SVD = TruncatedSVD(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        SVD.fit(np.concatenate([train_bow, test_bow]))
        train_svd = SVD.transform(train_bow)
        test_svd = SVD.transform(test_bow)
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mecabwordsvd200(AbstructFeature):
    DIMNUM = 200
    key = f"Mecabwordsvd{DIMNUM}"

    def create_features(self):
        global train_bow
        global test_bow
        SVD = TruncatedSVD(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        SVD.fit(np.concatenate([train_bow, test_bow]))
        train_svd = SVD.transform(train_bow)
        test_svd = SVD.transform(test_bow)
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mecabwordsvd300(AbstructFeature):
    DIMNUM = 300
    key = f"Mecabwordsvd{DIMNUM}"

    def create_features(self):
        global train_bow
        global test_bow
        SVD = TruncatedSVD(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        SVD.fit(np.concatenate([train_bow, test_bow]))
        train_svd = SVD.transform(train_bow)
        test_svd = SVD.transform(test_bow)
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mecabwordpca100(AbstructFeature):
    DIMNUM = 100
    key = f"Mecabwordpca{DIMNUM}"

    def create_features(self):
        global train_bow
        global test_bow
        pca = PCA(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        pca.fit(np.concatenate([train_bow, test_bow]))
        train_svd = pca.transform(train_bow)
        test_svd = pca.transform(test_bow)
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mecabwordpca200(AbstructFeature):
    DIMNUM = 200
    key = f"Mecabwordpca{DIMNUM}"

    def create_features(self):
        global train_bow
        global test_bow
        pca = PCA(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        pca.fit(np.concatenate([train_bow, test_bow]))
        train_svd = pca.transform(train_bow)
        test_svd = pca.transform(test_bow)
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mecabwordpca300(AbstructFeature):
    DIMNUM = 300
    key = f"Mecabwordpca{DIMNUM}"

    def create_features(self):
        global train_bow
        global test_bow
        pca = PCA(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        pca.fit(np.concatenate([train_bow, test_bow]))
        train_svd = pca.transform(train_bow)
        test_svd = pca.transform(test_bow)
        del train_bow
        del test_bow
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mecabboc(AbstructFeature):
    def create_features(self):
        global corpus_bow
        CCV = CountVectorizer(max_features=20000, ngram_range=(1, 3), analyzer="char")
        CCV.fit(corpus_bow)
        del corpus_bow
        global train_boc
        train_boc = CCV.transform(train["mecab_text"].values).toarray().tolist()
        self._train["Mecabboc"] = vec_to_vecstrs(train_boc)
        global test_boc
        test_boc = CCV.transform(test["mecab_text"].values).toarray().tolist()
        self._test["Mecabbow"] = vec_to_vecstrs(test_boc)


class Mecabchartfidf(AbstructFeature):
    def create_features(self):
        global train_boc
        global test_boc
        TFIDF = TfidfTransformer(sublinear_tf=True)
        TFIDF.fit(np.concatenate([train_boc, test_boc]))
        train_tfidf = TFIDF.transform(train_boc).toarray().tolist()
        test_tfidf = TFIDF.transform(test_boc).toarray().tolist()
        self._train["Mecabchartfidf"] = vec_to_vecstrs(train_tfidf)
        self._test["Mecabchartfidf"] = vec_to_vecstrs(test_tfidf)


class Mecabcharsvd100(AbstructFeature):
    DIMNUM = 100
    key = f"Mecabcharsvd{DIMNUM}"

    def create_features(self):
        global train_boc
        global test_boc
        SVD = TruncatedSVD(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        SVD.fit(np.concatenate([train_boc, test_boc]))
        train_svd = SVD.transform(train_boc)
        test_svd = SVD.transform(test_boc)
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mecabcharsvd200(AbstructFeature):
    DIMNUM = 200
    key = f"Mecabcharsvd{DIMNUM}"

    def create_features(self):
        global train_boc
        global test_boc
        SVD = TruncatedSVD(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        SVD.fit(np.concatenate([train_boc, test_boc]))
        train_svd = SVD.transform(train_boc)
        test_svd = SVD.transform(test_boc)
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mecabcharsvd300(AbstructFeature):
    DIMNUM = 300
    key = f"Mecabcharsvd{DIMNUM}"

    def create_features(self):
        global train_boc
        global test_boc
        SVD = TruncatedSVD(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        SVD.fit(np.concatenate([train_boc, test_boc]))
        train_svd = SVD.transform(train_boc)
        test_svd = SVD.transform(test_boc)
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mecabcharpca100(AbstructFeature):
    DIMNUM = 100
    key = f"Mecabcharpca{DIMNUM}"

    def create_features(self):
        global train_boc
        global test_boc
        pca = PCA(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        pca.fit(np.concatenate([train_boc, test_boc]))
        train_svd = pca.transform(train_boc)
        test_svd = pca.transform(test_boc)
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mecabcharpca200(AbstructFeature):
    DIMNUM = 200
    key = f"Mecabcharpca{DIMNUM}"

    def create_features(self):
        global train_boc
        global test_boc
        pca = PCA(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        pca.fit(np.concatenate([train_boc, test_boc]))
        train_svd = pca.transform(train_boc)
        test_svd = pca.transform(test_boc)
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mecabcharpca300(AbstructFeature):
    DIMNUM = 300
    key = f"Mecabcharpca{DIMNUM}"

    def create_features(self):
        global train_boc
        global test_boc
        pca = PCA(self.DIMNUM, random_state=RANDOM_STATE)
        SCL = StandardScaler()
        pca.fit(np.concatenate([train_boc, test_boc]))
        train_svd = pca.transform(train_boc)
        test_svd = pca.transform(test_boc)
        del train_boc
        del test_boc
        SCL.fit(np.concatenate([train_svd, test_svd]))
        train_scl = SCL.transform(train_svd).tolist()
        test_scl = SCL.transform(test_svd).tolist()
        self._train[self.key] = vec_to_vecstrs(train_scl)
        self._test[self.key] = vec_to_vecstrs(test_scl)


class Mcountuniquewords(AbstructFeature):
    def create_features(self):
        self._train["Mcountuniquewords"] = train["text"].apply(
            lambda x: len((set(x.split(" "))))
        )
        self._test["Mcountuniquewords"] = test["text"].apply(
            lambda x: len((set(x.split(" "))))
        )


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


class Keywordtopic5(AbstructFeature):
    """keyword単位でツイート集合を作ってトピックモデリングからベクトルを取得

    Args:
        AbstructFeature ([type]): [description]
    """

    DIMNUM = 5
    key = f"Keywordtopic{DIMNUM}"

    def create_features(self):
        topic_texts_pairs = defaultdict(list)
        for k in TOPIC_KEYWORDS.keys():
            topic_texts_pairs[k] = train[train.topic_keyword == k].mecab_text.to_list()
            topic_texts_pairs[k] += test[test.topic_keyword == k].mecab_text.to_list()
        cv = CountVectorizer(min_df=0.04, stop_words=STOPWORDS)
        lda = LDA(self.DIMNUM, random_state=RANDOM_STATE)
        corpus = [" ".join(v) for k, v in topic_texts_pairs.items()]
        cv.fit(corpus)
        cvec = cv.transform(corpus)
        lda.fit(cvec)
        lda_vec = lda.transform(cvec)
        str_lda_vec_list = []
        for l in lda_vec.tolist():
            str_lda_vec_list.append([str(val) for val in l])
        for i, k in enumerate(TOPIC_KEYWORDS.keys()):
            train.loc[
                train.topic_keyword == k, self.key
            ] = " ".join(str_lda_vec_list[i])
            test.loc[
                test.topic_keyword == k, self.key
            ] = " ".join(str_lda_vec_list[i])
        self._train[self.key] = train[self.key]
        self._test[self.key] = test[self.key]


class Keywordtopic10(AbstructFeature):
    """keyword単位でツイート集合を作ってトピックモデリングからベクトルを取得

    Args:
        AbstructFeature ([type]): [description]
    """

    DIMNUM = 10
    key = f"Keywordtopic{DIMNUM}"

    def create_features(self):
        topic_texts_pairs = defaultdict(list)
        for k in TOPIC_KEYWORDS.keys():
            topic_texts_pairs[k] = train[train.topic_keyword == k].mecab_text.to_list()
            topic_texts_pairs[k] += test[test.topic_keyword == k].mecab_text.to_list()
        cv = CountVectorizer(min_df=0.04, stop_words=STOPWORDS)
        lda = LDA(self.DIMNUM, random_state=RANDOM_STATE)
        corpus = [" ".join(v) for k, v in topic_texts_pairs.items()]
        cv.fit(corpus)
        cvec = cv.transform(corpus)
        lda.fit(cvec)
        lda_vec = lda.transform(cvec)
        str_lda_vec_list = []
        for l in lda_vec.tolist():
            str_lda_vec_list.append([str(val) for val in l])
        for i, k in enumerate(TOPIC_KEYWORDS.keys()):
            train.loc[
                train.topic_keyword == k, self.key
            ] = " ".join(str_lda_vec_list[i])
            test.loc[
                test.topic_keyword == k, self.key
            ] = " ".join(str_lda_vec_list[i])
        self._train[self.key] = train[self.key]
        self._test[self.key] = test[self.key]


class Keywordtopic20(AbstructFeature):
    """keyword単位でツイート集合を作ってトピックモデリングからベクトルを取得

    Args:
        AbstructFeature ([type]): [description]
    """

    DIMNUM = 20
    key = f"Keywordtopic{DIMNUM}"

    def create_features(self):
        topic_texts_pairs = defaultdict(list)
        for k in TOPIC_KEYWORDS.keys():
            topic_texts_pairs[k] = train[train.topic_keyword == k].mecab_text.to_list()
            topic_texts_pairs[k] += test[test.topic_keyword == k].mecab_text.to_list()
        cv = CountVectorizer(min_df=0.04, stop_words=STOPWORDS)
        lda = LDA(self.DIMNUM, random_state=RANDOM_STATE)
        corpus = [" ".join(v) for k, v in topic_texts_pairs.items()]
        cv.fit(corpus)
        cvec = cv.transform(corpus)
        lda.fit(cvec)
        lda_vec = lda.transform(cvec)
        str_lda_vec_list = []
        for l in lda_vec.tolist():
            str_lda_vec_list.append([str(val) for val in l])
        for i, k in enumerate(TOPIC_KEYWORDS.keys()):
            train.loc[
                train.topic_keyword == k, self.key
            ] = " ".join(str_lda_vec_list[i])
            test.loc[
                test.topic_keyword == k, self.key
            ] = " ".join(str_lda_vec_list[i])
        self._train[self.key] = train[self.key]
        self._test[self.key] = test[self.key]


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

    variables = {k: v for k, v in variables.items() if is_AbstructFeature(v)}
    generate_features(variables, args.force)
