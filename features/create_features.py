import os
import re
import string
import unicodedata
from collections import defaultdict
from typing import List, Union

import MeCab
import neologdn
import pandas as pd

from base import AbstructFeature, generate_features, get_arguments
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
        self._train["Keyword"] = train["keyword"]
        self._test["Keyword"] = test["keyword"]


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
        self._train["Nkeyword"] = train["keyword"].apply(self.replace_rule)
        self._test["Nkeyword"] = test["keyword"].apply(self.replace_rule)
        train["nkeyword"] = self._train["Nkeyword"]
        test["nkeyword"] = self._test["Nkeyword"]

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
        self._train["Id"] = (train.id - train.id.mean()) / (
            train.id.max() - train.id.min()
        )
        self._test["Id"] = (test.id - test.id.mean()) / (test.id.max() - test.id.min())


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
    ASCII_LOWERCASES = [c for c in "abcdefghijklmnopqrstuvwxyz"]

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
            if (
                node.feature.split(",")[0] in ["名詞", "動詞", "形容詞", "副詞"]
                and len(node.surface) > 1
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
