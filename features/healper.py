import re
import emoji
import pandas as pd


RESTRDIC = {
    "url": r"https?://[\/\w\:%#$&\?\(\)!\.=\+\-]+",
    "hashtag": r"#\S+",
    "kaomoji": r"\([^あ-ん\u30A1-\u30F4\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+?\)",
    "number": r"\d+",
}

DUP_AND_LMISS_DIC = {
    "JR東海道線であわや大惨事 崩落現場からわずか100m手前でストップ #ldnews https://t.co/St6BSmNitE": 0,
    "武漢から退避するドイツ救援機の着陸許さず ロシアへ反発も。ドイツ国防相は1日、中国・武漢から帰国の途についた空軍救援機に言及。乗員交代のために計画露モスクワでの一時着陸が許可されなかった。空港の能力不足が理由とされるが、ドイツ政界からはロシアへの反発もある( ･ω･)": 0,
    "◆鹿児島市星ヶ峯の市道で､市街地方面に向かっていた普通乗用車が対向車線をはみ出し､直進してきたﾊﾞｲｸと正面衝突した｡この事故でﾊﾞｲｸに乗っていた畠中正典さん５２歳が頭などを強く打ち､搬送先の病院で死亡した｡畠中さんは新聞配達のｱﾙﾊﾞｲﾄ中でした｡": 1,
    "北海道・長野で雪崩相次ぐ、英国人ら巻き込まれ死亡。(・｀ω´・)": 1,
}


url = re.compile(RESTRDIC["url"])
hashtag = re.compile(RESTRDIC["hashtag"])
kaomoji = re.compile(RESTRDIC["kaomoji"])
number = re.compile(RESTRDIC["number"])


def in_reply(text: str):
    return len([c for c in text if c == "@"])


def in_url(text: str):
    return len(url.findall(text))


def in_hashtag(text: str):
    return len(hashtag.findall(text))


def in_emoji(text: str):
    return len([c for c in text if c in emoji.UNICODE_EMOJI])


def replace_emoji(text: str):
    replaced_text = ""
    for c in text:
        if c in emoji.UNICODE_EMOJI:
            replaced_text += " "
        else:
            replaced_text += c
    return replaced_text


def in_kaomoji(text: str):
    return len(kaomoji.findall(text))


def in_number(text: str):
    return len(number.findall(text))


def replace_reply(text: str):
    return re.sub(r"@[A-Za-z0-9_]+", "", text)


def count_date_representation(text: str):
    return len(re.findall(r"(\d年|\d月|\d日|\d時|\d分|\d秒)", text))


def load_csv_for_train_to_df(path: str):
    df = pd.read_csv(path)
    # 重複しているかつラベルが違うもののラベルを修正
    for k, v in DUP_AND_LMISS_DIC.items():
        df.loc[df.text == k, "flg"] = v
    # 重複しているテキストを1つに集約
    df = df.drop_duplicates(subset=["text", "location"]).reset_index(drop=True)
    df.flg.to_csv("./data/4.true_flg.csv")
    return df
