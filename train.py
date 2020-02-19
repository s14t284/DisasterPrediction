import numpy as np
import pandas as pd
import MeCab
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


RANDOM_STATE = 0

df_train = pd.read_csv("./data/1.train_data.csv")
df_test = pd.read_csv("./data/2.test_data.csv")

tagger = MeCab.Tagger("-Owakati")

stop_words = set(["の", "に", "て", "、", "が", "た", "。", "は", "を", "で"])
with open("data/slothlib.txt", "r") as f:
    # stop_words = set([line.replace("\n", "") for line in f.readlines()])
    print(stop_words)

train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=RANDOM_STATE)

train_df["text"] = train_df["text"].apply(tagger.parse)
train_df = train_df.reset_index(drop=True)
val_df["text"] = val_df["text"].apply(tagger.parse)
val_df = val_df.reset_index(drop=True)

cv = CountVectorizer()
cv.fit(train_df["text"].values)

X_train = cv.transform(train_df["text"].values)
y_train = train_df["flg"].values

X_val = cv.transform(val_df["text"].values)
y_val = val_df["flg"].values

# l_svc = XGBClassifier(random_state=RANDOM_STATE)
l_svc = LinearSVC(random_state=RANDOM_STATE)
l_svc.fit(X_train, y_train)
print(classification_report(y_val, l_svc.predict(X_val)))

df_test["text"] = df_test["text"].apply(tagger.parse)
X_test = cv.transform(df_test["text"].values)

predict = l_svc.predict(X_test)
submission = pd.read_csv("./data/3.submission.csv")
submission["flg"] = predict
submission.to_csv("submission.csv", index=False)
