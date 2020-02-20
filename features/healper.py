import re
import emoji


RESTRDIC = {
    "url": r"https?://[\w\:%#$&\?\(\)!\.=\+\-]+",
    "hashtag": r"#.+",
    "kaomoji": r"\([^あ-ん\u30A1-\u30F4\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+?\)",
    "number": r"\d+",
}


url = re.compile(RESTRDIC["url"])
hashtag = re.compile(RESTRDIC["hashtag"])
kaomoji = re.compile(RESTRDIC["kaomoji"])
number = re.compile(RESTRDIC["number"])


def in_url(text: str):
    return url.search(text) is not None


def in_hashtag(text: str):
    return hashtag.search(text) is not None


def in_emoji(text: str):
    for c in text:
        if c in emoji.UNICODE_EMOJI:
            return True
    return False


def replace_emoji(text: str):
    replaced_text = ""
    for c in text:
        if c in emoji.UNICODE_EMOJI:
            replaced_text += " "
        else:
            replaced_text += c
    return replaced_text


def in_kaomoji(text: str):
    return kaomoji.search(text) is not None


def in_number(text: str):
    return number.search(text) is not None


def replace_reply(text: str):
    return re.sub(r"@[A-Za-z0-9_]+", "", text)
